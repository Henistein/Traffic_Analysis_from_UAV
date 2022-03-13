import fiftyone.zoo as foz
import torch
from torch.utils.data import Dataset, DataLoader
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
  def __init__(self, imgs_path, labels_path=None):
    self.imgs_path = imgs_path
    self.labels_path = labels_path
    self.images = glob.glob(imgs_path + "/*")
    self.labels = glob.glob(labels_path + "/*")
    assert len(self.images) == len(self.labels), "Number of labels and images differ"
    self.names = [Path(f).stem for f in self.images]
    # preprocess settings
    self.classes = [3, 4, 5, 6, 7, 8, 9, 10]
    self.imsize = 640
    self.transforms = tfs.Compose([
        tfs.Resize((self.imsize, self.imsize)),
        tfs.ToTensor(),
    ])

  
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    name = self.names[index]
    img_file = glob.glob(self.imgs_path+f"/{name}.*")
    label_file = glob.glob(self.labels_path+f"/{name}.*")

    assert len(img_file) == len(label_file), "There is labels or images with the same name"
    img_file, label_file = img_file[0], label_file[0]

    # preprocess image 
    img = Image.open(img_file)
    w, h = img.size
    img = self.transforms(img)

    # preprocess label
    f_aux = open(label_file)
    label = []
    for line in f_aux.readlines():
      data = line.split(',')
      # check if class is filtered
      cls = int(data[0])
      assert cls in self.classes, "Classes not filtered"
      cls -= 3 # convert it to fit between 0 and 7

      bbox = list(map(float, data[:4]))

      # normalize bbox and scale it with imsize
      bbox[0] = (bbox[0] / w) * self.imsize
      bbox[1] = (bbox[1] / h) * self.imsize
      bbox[2] = (bbox[2] / w) * self.imsize
      bbox[3] = (bbox[3] / h) * self.imsize

      label.append([0]+[cls]+bbox)
    
    label = torch.tensor(label)

    return img, label
  
