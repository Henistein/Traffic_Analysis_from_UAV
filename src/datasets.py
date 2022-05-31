#import fiftyone.zoo as foz
import torch
import cv2
import glob
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from utils.general import letterbox, image_loader
from utils.conversions import xywhn2xyxy, xywhn2xyxy, xyxy2xywhn, xywh2xyxy

class MyDataset(Dataset):
  def __init__(self, imgs_path, labels_path, imsize=640, DET=False):
    self.imgs_path = imgs_path
    self.labels_path = labels_path
    self.images = glob.glob(imgs_path + "/*")
    self.labels = np.loadtxt(labels_path, delimiter=',') if not DET else glob.glob(labels_path+"/*")
    if not DET:
      assert len(self.images) == self.labels[-1, ..., 0], "Number of labels and images differ"
    else:
      assert len(self.images) == len(self.labels), "Number of labels and images differ"
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

  def convert_box(self, size, box):
    # Convert VisDrone box to YOLO xywh box
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh
  
  def __len__(self):
    return len(self.names)
  
class MyDatasetMOT(MyDataset):
  def __init__(self, imgs_path, labels_path, imsize=640):
    super().__init__(imgs_path, labels_path, imsize)

  def __getitem__(self, index):
    name = self.names[index]
    img_file = glob.glob(self.imgs_path+f"/{name}.*")[0]
    
    # load image 
    img = cv2.imread(img_file)
    img, h0, w0, ratio, pad = image_loader(img, self.imsize, True)
    img = img[0]
    h, w = img.shape[1:]

    shapes = (h0, w0), ((h/h0, w/w0), pad)

    # load labels
    label_id = int(name.split('_')[1]) + 1
    labels = self.labels[self.labels[:, 0] == label_id]

    # ids [x y w h] cls
    ids = labels[..., 1]
    bbox = labels[..., 2:6]
    cls = labels[..., 7] - 1

    # scale to native (default image shape)
    bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]/2) #/ w0
    bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]/2) #/ h0
    bbox = xywh2xyxy(bbox)

    # concatenate 
    labels_out = torch.zeros((len(labels), 6))
    labels_out[:, 0] = torch.from_numpy(ids)
    labels_out[:, 1:5] = torch.from_numpy(bbox)
    labels_out[:, 5] = torch.from_numpy(cls)

    # filter ingored regions
    labels_out = labels_out[labels_out[:, 5] != -1]

    img = img.unsqueeze(0)

    return img, labels_out, img_file, shapes

class MyDatasetDET(MyDataset):
  def __init__(self, imgs_path, labels_path, imsize=640):
    super().__init__(imgs_path, labels_path, imsize, DET=True)

  def __getitem__(self, index):
    name = self.names[index]
    img_file = glob.glob(self.imgs_path+f"/{name}.*")[0]
    label_file = glob.glob(self.labels_path+f"/{name}.*")[0]
    
    # load image 
    img = cv2.imread(img_file)
    img, h0, w0, ratio, pad = image_loader(img, self.imsize, True)
    img = img[0]
    h, w = img.shape[1:]

    shapes = (h0, w0), ((h/h0, w/w0), pad)

    # load label
    labels = np.loadtxt(label_file, delimiter=',')
    if labels.ndim == 1:
      labels = labels.reshape(1, -1)

    # [x y w h] cls
    bbox = labels[..., :4]
    cls = labels[..., 5] - 1

    # scale to native (default image shape)
    bbox[:, 0] = (bbox[:, 0] + bbox[:, 2]/2) #/ w0
    bbox[:, 1] = (bbox[:, 1] + bbox[:, 3]/2) #/ h0
    bbox = xywh2xyxy(bbox)

    # concatenate 
    labels_out = torch.zeros((len(labels), 5))
    labels_out[:, :4] = torch.from_numpy(bbox)
    labels_out[:, 4] = torch.from_numpy(cls)

    # filter ingored regions
    labels_out = labels_out[labels_out[:, 4] != -1]

    img = img.unsqueeze(0)

    return img, labels_out, img_file, shapes
