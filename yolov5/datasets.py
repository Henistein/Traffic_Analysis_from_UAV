#import fiftyone.zoo as foz
import torch
import cv2
import glob
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from utils.general import letterbox
from utils.conversions import xywhn2xyxy, xywhn2xyxy, xyxy2xywhn

class MyDataset(Dataset):
  def __init__(self, imgs_path, labels_path, imsize=640):
    self.imgs_path = imgs_path
    self.labels_path = labels_path
    self.images = glob.glob(imgs_path + "/*")
    self.labels = np.loadtxt(labels_path, delimiter=',')
    assert len(self.images) == self.labels[-1, ..., 0], "Number of labels and images differ"
    self.names = [Path(f).stem for f in self.images]
    self.names.sort()
    # preprocess settings
    self.imsize = imsize

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
    img_file = glob.glob(self.imgs_path+f"/{name}.*")[0]
    
    # load image 
    img, (h0, w0), (h, w) = self.load_image(img_file)

    img, ratio, pad = letterbox(img, self.imsize, auto=False)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)

    shapes = (h0, w0), ((h/h0, w/w0), pad)

    # load labels
    label_id = int(name.split('_')[1]) + 1
    labels = self.labels[self.labels[:, 0] == label_id]

    # ids [x y w h] cls
    ids = labels[..., 1]
    bbox = labels[..., 2:6]
    cls = labels[..., -2] - 1

    # normalize 
    bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]/2) / w0
    bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]/2) / h0
    bbox[:, 2] /= w0
    bbox[:, 3] /= h0

    # normalized xywh to pixel xyxy format
    bbox = xywhn2xyxy(bbox, ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
    # pixel xyxy format to xywh normalized
    bbox = xyxy2xywhn(bbox, w=img.shape[1], h=img.shape[2], clip=True, eps=1E-3)

    # concatenate 
    labels_out = torch.zeros((len(labels), 6))
    labels_out[:, 0] = torch.from_numpy(ids)
    labels_out[:, 1:5] = torch.from_numpy(bbox)
    labels_out[:, 5] = torch.from_numpy(cls)

    img = img.unsqueeze(0)

    return img, labels_out, img_file, shapes