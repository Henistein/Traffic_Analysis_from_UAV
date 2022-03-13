import torch
import torch.nn as nn
from torch.cuda import amp
from models.yolo import Model
import yaml
from utils.loss import ComputeLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import VisdroneDataset


# some auxiliar functions change after done
def de_parallel(model):
  return model.module if is_parallel(model) else model

def is_parallel(model):
  # Returns True if model is of type DP or DDP
  return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


mydataset = VisdroneDataset(
  imgs_path='/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone/VisDrone2019-DET-train/images',
  labels_path='/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone/VisDrone2019-DET-train/annotations'
)
loader = DataLoader(mydataset, batch_size=1)


# cfg
cfg = 'cfg.yaml'
device = torch.device('cpu' if torch.cuda.is_available() else 'cuda')
cuda = device.type == 'cuda'
nc = 8
imgsz = 640

# Model
model = Model(cfg, ch=3, nc=nc)

# Model atributes
hyp = yaml.safe_load(open('hyp.yaml'))
nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
hyp['box'] *= 3 / nl  # scale to layers
hyp['cls'] *= nc / 8 * 3 / nl  # scale to classes and layers
hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
hyp['label_smoothing'] = 0
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
optimizer = torch.optim.Adam(g0, lr, betas=(momentum, 0.999))
lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# train


for epoch in range(epochs):
  model.train()
  pbar = tqdm(enumerate(loader), total=len(loader))
  for i,(imgs,targets) in pbar:
    imgs = imgs.to(device).float()
    print(targets.shape)

    # Forward
    with amp.autocast(enabled=cuda):
      pred = model(imgs) # forward
      targets = targets.squeeze(0)

      loss, loss_items = compute_loss(pred, targets)
    
    # Backward
    scaler.scale(loss).backward()
    
    # Optimize
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # update bar
    pbar.set_description(f"Loss: {loss.item()}")

  # Scheduler 
  scheduler.step()



#print("AUQI")
#sample = dataset[0]
#labels = [[0.0]+[classnames[det['label']]]+det['bounding_box'] for det in sample['ground_truth']['detections']]
#labels = torch.tensor(labels)
#print(labels)
#exit(0)








