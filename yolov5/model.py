import torch
import torch.nn as nn
import pkg_resources as pkg

def autopad(k, p=None):  # kernel, padding
  # Pad to 'same'
  if p is None:
    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
  return p

class Conv(nn.Module):
  # Standard convolution
  def __init__(self, ch_in, ch_out, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
    super().__init__()
    self.conv = nn.Conv2d(ch_in, ch_out, k, s, autopad(k, p), groups=g, bias=False)
    self.bn = nn.BatchNorm2d(ch_out)
    self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
  # Standard bottleneck
  # ch_in, ch_out, shortcut, groups, expansion
  def __init__(self, ch_in, ch_out, shortcut=True, g=1, e=0.5):
    super().__init__()
    c_ = int(ch_in * e)  # hidden channels
    self.cv1 = Conv(ch_out, c_, 1, 1)
    self.cv2 = Conv(c_, ch_out, 3, 1, g=g)
    self.add = shortcut and ch_in == ch_out

  def forward(self, x):
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Concat(nn.Module):
  # Concatenate a list of tensors along dimension
  def __init__(self, dimension=1):
    super().__init__()
    self.d = dimension

  def forward(self, x):
    return torch.cat(x, self.d)

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

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by YOLOv3, but {name}{current} is currently installed'
    else:
        return result


IN_CH = 3
M = nn.Sequential(
    Conv(IN_CH, 32, 3, 1),
    Conv(32, 64, 3, 2),
    Bottleneck(64, 64),
    Conv(64, 128, 3, 2),

    Bottleneck(128, 128),
    Bottleneck(128, 128),
    Conv(128, 256, 3, 2),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Bottleneck(256, 256),
    Conv(256, 512, 3, 2),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Bottleneck(512, 512),
    Conv(512, 1024, 3, 2),
    Bottleneck(1024, 1024),
    Bottleneck(1024, 1024),
    Bottleneck(1024, 1024),
    Bottleneck(1024, 1024),
)
print(M)