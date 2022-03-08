from copy import deepcopy
import torch
import torch.nn as nn
import yaml

from models.common import *

torch.manual_seed(420)


def parse_model(d, ch):  # model_dict, input_channels(3)
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    print('AQUI: ', anchors)
    na = (len(anchors[0]) // 2) # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, C3, SPPF]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m is C3:
              args.insert(2, n)  # number of repeats
              n = 1
            
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        m_.i, m_.f = i, f # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(
            anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(
                        nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 +
                                   self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                        self.anchor_grid[i]  # wh
                else:  # for  on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * \
                        self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid([torch.arange(ny).to(
                d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid(
                [torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

class Model(nn.Module):
  def __init__(self, cfg, ch=3):
    super().__init__()
    self.yaml = yaml.safe_load(open(cfg))
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=[3])
    self.inplace = True

    # Build strides, anchors
    m = self.model[-1]  # Detect()
    if isinstance(m, Detect):
      s = 256  # 2x min stride
      m.inplace = self.inplace
      m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
      m.anchors /= m.stride.view(-1, 1, 1)
      self.stride = m.stride
      self._initialize_biases()  # only run once

    # Init weights, biases
    for mdl in self.modules():
      t = type(mdl)
      if t is nn.BatchNorm2d:
        mdl.eps = 1e-3
        mdl.momentum = 0.03
      elif t is nn.SiLU:
        mdl.inplace = True
    
  def forward(self, x):
    y = []  # outputs
    for m in self.model:
      if m.f != -1:  # if not from previous layer
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
      x = m(x)  # run
      y.append(x if m.i in self.save else None)  # save output
    return x
  
  def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
    # https://arxiv.org/abs/1708.02002 section 3.3
    # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    m = self.model[-1]  # Detect() module
    for mi, s in zip(m.m, m.stride):  # from
      b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
      b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
      b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
      mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

"""
if __name__ == "__main__":
  # load yaml config
  #y = yaml.safe_load(open('cfg.yaml'))
  #y['ch'] = 3
  #model1, save1 = parse_model(deepcopy(y), ch=[3])

  model = Model('cfg.yaml')
  model.train()

  img = torch.rand(1, 3, 416, 416).float()
  pred = model(img)[0]
  print(pred.shape)

  pred = non_max_suppression(pred)[0]
  print(pred)
"""