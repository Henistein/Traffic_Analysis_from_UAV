import fiftyone.zoo as foz
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from utils.general import letterbox
from PIL import Image
import torchvision.transforms as tfs

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

class VisdroneDataset(Dataset):
  def __init__(self, imgs_path, labels_path=None, samples=None):
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
    self.classes = [3, 4, 5, 6, 7, 8, 9, 10]
    self.imsize = 640
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
    label_file = glob.glob(self.labels_path+f"/{name}.*")

    assert len(img_file) == len(label_file), "There is labels or images with the same name"
    img_file, label_file = img_file[0], label_file[0]

    # load image 
    img, (h0, w0), (h, w) = self.load_image(img_file)
    img, ratio, pad = letterbox(img, self.imsize, auto=False)
    img = self.transforms(img)
    shapes = (h0, w0), ((h/h0, w/w0), pad)

    # load labels
    f_aux = open(label_file)
    label = []
    for line in f_aux.readlines():
      data = line.split(',')
      # check if class is filtered
      #assert cls in self.classes, "Classes not filtered"
      #cls -= 3 # convert it to fit between 0 and 7

      cls = int(data[0])
      bbox = list(map(int, data[:4]))

      # convert visdrone format to yolo format
      bbox = self.convert_box((w0, h0), bbox)

      label.append([0]+[cls]+list(bbox))
    
    label = torch.tensor(label)

    return img, label, img_file, shapes
