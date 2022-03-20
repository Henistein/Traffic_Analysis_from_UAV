import torch
import torch.nn as nn
from torch.cuda import amp
from models.yolo import Model
import yaml
import numpy as np
from utils.loss import ComputeLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import VisdroneDataset
from inference import get_pred, compute_stats
from utils.metrics import process_batch
from utils.general import non_max_suppression
from utils.conversions import scale_coords, xywh2xyxy, xywhn2xyxy, xywhn2xyxy
import math
from copy import deepcopy


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    """
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
    """



# some auxiliar functions change after done
def de_parallel(model):
  return model.module if is_parallel(model) else model

def is_parallel(model):
  # Returns True if model is of type DP or DDP
  return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

batch_size = 10
def load_visdrone_datasets():

  t = VisdroneDataset(
    imgs_path='/home/henistein/playground/datasets/VisDrone/VisDrone2019-DET-train/images',
    labels_path='/home/henistein/playground/datasets/VisDrone/VisDrone2019-DET-train/annotations'
  )
  tt = DataLoader(t,
                  batch_size=batch_size,
                  pin_memory=True,
                  collate_fn=VisdroneDataset.collate_fn)

  v = VisdroneDataset(
    imgs_path='/home/henistein/playground/datasets/VisDrone/VisDrone2019-DET-val/images',
    labels_path='/home/henistein/playground/datasets/VisDrone/VisDrone2019-DET-val/annotations'
  )
  vv = DataLoader(v,
                  batch_size=2,
                  pin_memory=True,
                  collate_fn=VisdroneDataset.collate_fn)

  return tt, vv
@torch.no_grad()
def validation_round(model, val_dataset):
  model.eval()
  pbar = tqdm(enumerate(val_dataset), total=len(val_dataset))
  stats = []
  iou = torch.linspace(0.5, 0.95, 10)
  for i,(im,targets,paths,shapes) in pbar:
    im = im.to(device, non_blocking=True)
    targets = targets.to(device)
    im = im.float() / 255
    nb, _, height, width = im.shape

    # Inference
    out = model(im)[0]

    # NMS
    targets[:, 2:] *= torch.tensor((width, height, width, height), device=device) # to pixels
    out = non_max_suppression(out, 0.001, 0.6)

    # Metrics
    for si, pred in enumerate(out):
      labels = targets[targets[:, 0] == si, 1:]
      nl = len(labels)
      tcls = labels[:, 0].tolist() if nl else []
      shape = shapes[si][0]

      if len(pred) == 0:
        if nl:
          stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
          continue 

      # Predictions
      predn = pred.clone()
      scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1]) # natice-space pred

      # Evaluate
      if nl:
        tbox = xywh2xyxy(labels[:, 1:5]) # target boxes 
        scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
        labelsn = torch.cat((labels[:, 0:1], tbox), 1)
        correct = process_batch(predn, labelsn, iou)

      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)


  # compute stats
  results = compute_stats(stats)

  return results



# cfg
cfg = 'cfg.yaml'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = device.type == 'cuda'
nc = 10
imgsz = 640

# Model
#model.load_state_dict(torch.load('weights/visdrone.pth'))
model = Model(cfg, ch=3, nc=nc).to(device)
# EMA
ema = ModelEMA(model)

# Hyperparameters
nbs = 64 # nominal batch
accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
hyp = yaml.safe_load(open('hyp.yaml'))
hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
hyp['box'] *= 3 / nl  # scale to layers
hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
hyp['label_smoothing'] = 0

model.nc = nc  # attach number of classes to model
model.hyp = hyp  # attach hyperparameters to model

compute_loss = ComputeLoss(model)
scaler = amp.GradScaler(enabled=cuda)

g0, g1, g2 = [], [], []  # optimizer parameter groups
for v in model.modules():
  if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
    g2.append(v.bias)
  if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
    g0.append(v.weight)
  elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
    g1.append(v.weight)

epochs = 100
lr = 0.01
momentum = 0.937
# Optimizer
optimizer = torch.optim.SGD(g0, lr=lr, momentum=momentum, nesterov=True)
optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
optimizer.add_param_group({'params': g2})  # add g2 (biases)

lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
scheduler.last_epoch = -1  # do not move

train_loader, val_loader = load_visdrone_datasets()
# train
nb = len(train_loader)
nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
last_opt_step = -1

for epoch in range(epochs):
  model.train()
  pbar = tqdm(enumerate(train_loader), total=len(train_loader))
  optimizer.zero_grad()
  for i,(imgs,targets,paths, _) in pbar:
    ni = i + nb * epoch  # number integrated batches (since train start)
    imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
    

    # Warmup
    if ni <= nw:
      xi = [0, nw]  # x interp
      # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
      accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
      for j, x in enumerate(optimizer.param_groups):
        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
        if 'momentum' in x:
          x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
      

    # Forward
    with amp.autocast(enabled=cuda):
      pred = model(imgs) # forward
      loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

    # Backward
    scaler.scale(loss).backward()
    
    # Optimizer
    if ni - last_opt_step >= accumulate:
      scaler.step(optimizer)  # optimizer.step
      scaler.update()
      optimizer.zero_grad()
      if ema:
        ema.update(model)
      last_opt_step = ni

    # update bar
    pbar.set_description(f"Loss: {loss.item()}")
  

  # Scheduler
  scheduler.step()

  # validation round
  results = validation_round(ema.ema, val_loader)
  print('Results: ', results)


torch.save(model.state_dict(), 'weights/visdrone.pth')

