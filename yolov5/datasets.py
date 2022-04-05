import fiftyone.zoo as foz
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from utils.general import letterbox
from PIL import Image
import torchvision.transforms as tfs
from utils.conversions import scale_coords, xywh2xyxy, xywhn2xyxy, xyxy2xywhn
from utils.metrics import process_batch

def load_coco_classnames():
  classnames = [cl.strip() for cl in open('coco_classes.txt').readlines()]
  classnames = {k:classnames.index(k) for k in classnames}
  return classnames

# dataset
def load_coco_dataset():
  dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=100,
    shuffle=False,
  )

  dataset = list(dataset)
  return dataset

class SimpleCustomBatch:
  def __init__(self, data):
    transposed_data = list(zip(*data))
    self.inp = torch.stack(transposed_data[0], 0)
    self.tgt = torch.stack(transposed_data[1], 1)
  
  def pin_memory(self):
    self.inp = self.inp.pin_memory()
    self.tgt = self.tgt.pin_memory()
    return self

def collate_wrapper(batch):
  return SimpleCustomBatch(batch)

class MyDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = list(dataset)
    self.classnames = load_coco_classnames()
    self.transforms = tfs.Compose([
        tfs.Resize((640, 640)),
        tfs.ToTensor(),
    ])

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    sample = self.dataset[index]

    # get labels
    labels = [[0.0]+[self.classnames[det['label']]]+det['bounding_box'] for det in sample['ground_truth']['detections']]
    labels = torch.tensor(labels)

    # get imgs
    img = Image.open(sample.filepath)
    # apply transforms
    img = self.transforms(img)

    return img, labels


# dataset for visdrone
import glob
#import os
from pathlib import Path
import numpy as np

import torchvision.transforms as tfs

class VisdroneDataset(Dataset):
  def __init__(self, imgs_path, labels_path, samples=None, imsize=640):
    self.imgs_path = imgs_path
    self.labels_path = labels_path
    self.images = glob.glob(imgs_path + "/*")
    self.labels = glob.glob(labels_path + "/*")
    assert len(self.images) == len(self.labels), "Number of labels and images differ"
    if samples:
      self.names = [Path(f).stem for f in self.images][:samples]
    else:
      self.names = [Path(f).stem for f in self.images]
    # preprocess settings
    self.imsize = imsize
    self.transforms = tfs.Compose([
        tfs.ToTensor(),
    ])

  @staticmethod
  def collate_fn(batch):
    img, label, path, shapes = zip(*batch)
    for i, lb in enumerate(label):
      lb[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes

  def load_image(self, img_path):
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2] # original w and h
    r = self.imsize / max(h0, w0) # ratio
    if r != 1: # if sizes are not equal
      img = cv2.resize(
           img,
           (int(w0*r), int(h0*r)),
           interpolation=cv2.INTER_LINEAR if r >1 else cv2.INTER_AREA
      )
    return img, (h0, w0), img.shape[:2] # img, hw_original, hw_resized
                       
  
  def convert_box(self, size, box):
    # Convert VisDrone box to YOLO xywh box
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh
  
  def __len__(self):
    return len(self.names)
  
  def __getitem__(self, index):
    name = self.names[index]
    img_file = glob.glob(self.imgs_path+f"/{name}.*")
    labels_file = glob.glob(self.labels_path+f"/{name}.*")

    assert len(img_file) == len(labels_file), "There is labels or images with the same name"
    img_file, labels_file = img_file[0], labels_file[0]

    # load image 
    img, (h0, w0), (h, w) = self.load_image(img_file)
    img, ratio, pad = letterbox(img, self.imsize, auto=False)
    img = self.transforms(img)
    shapes = (h0, w0), ((h/h0, w/w0), pad)

    # load labels
    f_aux = open(labels_file)
    labels = []
    for line in f_aux.readlines():
      data = line.split(' ')

      cls = int(data[0])
      bbox = list(map(float, data[1:]))

      # convert visdrone format to yolo format
      #bbox = self.convert_box((w0, h0), bbox)


      labels.append([0]+[cls]+list(bbox))
    
    labels = torch.tensor(labels)
    # normalized xywh to pixel xyxy format
    #labels[:, 2:] = xywhn2xyxy(labels[:, 2:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
    # pixel xyxy format to xywh normalized
    #labels[:, 2:] = xyxy2xywhn(labels[:, 2:], w=img.shape[1], h=img.shape[2], clip=True, eps=1E-3)

    return img, labels, img_file, shapes


from inference import Inference
from utils.general import non_max_suppression
from utils.conversions import scale_coords, xywh2xyxy, xywhn2xyxy, xywhn2xyxy
from tqdm import tqdm

if __name__ == '__main__':
  dataset = VisdroneDataset(
              imgs_path='/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone2019-DET-test-dev/images',
              labels_path='/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone2019-DET-test-dev/labels'
            )
  dataloader = DataLoader(dataset,
                  batch_size=4,
                  pin_memory=True,
                  collate_fn=VisdroneDataset.collate_fn)

  weights = 'weights/visdrone.pt'
  device = torch.device('cuda')
  model = torch.load(weights)['model'].float()
  model.to(device)


  # validation
  model.eval()
  stats = []
  iou = torch.linspace(0.5, 0.95, 10)

  for i,(img,targets,paths,shapes) in enumerate(tqdm(dataloader, total=len(dataloader))):
    img = img.to(device, non_blocking=True)
    if '/home/socialab/Henrique/datasets/VisDrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg' in paths:
      print(paths, img)
      exit(0)
    targets = targets.to(device)
    #img = img.float() / 255
    nb, _, height, width = img.shape

    # Inference
    out = model(img)[0]

    # NMS 
    targets[:, 2:] *= torch.tensor((width, height, width, height), device=device) # to pixels
    out = non_max_suppression(out, 0.001, 0.6)

    # Metrics
    for si, pred in enumerate(out):
      if si == 100: break
      labels = targets[targets[:, 0] == si, 1:]
      nl = len(labels)
      tcls = labels[:, 0].tolist() if nl else []
      shape = shapes[si][0]

      if pred is None or len(pred) == 0:
        if nl:
          stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
          continue 

      # Predictions
      predn = pred.clone()
      scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1]) # natice-space pred

      # Evaluate
      if nl:
        tbox = xywh2xyxy(labels[:, 1:5]) # target boxes 
        scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
        labelsn = torch.cat((labels[:, 0:1], tbox), 1)
        correct = process_batch(predn, labelsn, iou)

      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)


  # compute stats
  results = Inference.compute_stats(stats)
  print(results)
