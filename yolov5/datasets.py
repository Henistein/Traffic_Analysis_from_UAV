#import fiftyone.zoo as foz
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from utils.general import letterbox, image_loader
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


from inference import Inference, Annotator
from utils.general import non_max_suppression
from utils.conversions import scale_coords, xywh2xyxy, xywhn2xyxy, xywhn2xyxy
from tqdm import tqdm

if __name__ == '__main__':
  dataset = VisdroneDataset(
              imgs_path='/home/henistein/projects/ProjetoLicenciatura/datasets/rotunda2/images',
              labels_path='/home/henistein/projects/ProjetoLicenciatura/datasets/rotunda2/labels.txt'
            )
  """
  dataset = VisdroneDataset(
              imgs_path='/home/socialab/Henrique/datasets/VisDrone/VisDrone2019-DET-test-dev/images',
              labels_path='/home/socialab/Henrique/datasets/VisDrone/VisDrone2019-DET-test-dev/labels',
            )
  """
  """
  dataloader = DataLoader(dataset,
                  batch_size=1,
                  pin_memory=True,
                  collate_fn=VisdroneDataset.collate_fn)
  """

  weights = 'weights/visdrone5l.pt'
  #weights = 'weights/yolov5l-xs.pt'
  device = torch.device('cuda')
  model = torch.load(weights)['model'].float()
  model.to(device)


  # validation
  model.eval()
  classnames = model.names
  stats = []
  annotator = Annotator()
  iou = torch.linspace(0.5, 0.95, 10)

  for i,(img,targets,paths,shapes) in enumerate(tqdm(dataset, total=len(dataset))):
    img = img.to(device, non_blocking=True)
    img = img.float() / 255

    targets = targets.to(device)
    nb, _, height, width = img.shape

    # Inference
    out = model(img)[0]

    # NMS 
    targets[:, 1:5] *= torch.tensor((width, height, width, height), device=device) # to pixels
    out = non_max_suppression(out, 0.5, 0.5)

    # Metrics
    for si, pred in enumerate(out):
      bbox = targets[:, 1:5].clone()
      nl = len(bbox)
      tcls = targets[:, -1].tolist() if nl else []
      shape = shapes[0]

      if pred is None or len(pred) == 0:
        if nl:
          stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
          continue 
      
      # Predictions
      predn = pred.clone()
      scale_coords(img[0].shape[1:], predn[:, :4], shape, shapes[1]) # natice-space pred

      # Evaluate
      if nl:
        tbox = xywh2xyxy(bbox).cpu() # target boxes 
        scale_coords(img[0].shape[1:], tbox, shape, shapes[1])  # native-space labels

        cls_torch = torch.tensor(tcls).reshape(-1, 1)
        labelsn = torch.cat((cls_torch, tbox), 1)
        # visualize
        im = cv2.imread(paths)
        lcc = Inference.labels_conf_cls(labels=labelsn[:, 1:], conf=None, cls=labelsn[:, 0]) # labels conf cls format
        outimg = Inference.attach_detections(annotator, lcc, im, classnames, is_label=True)

        correct = process_batch(predn.cpu(), labelsn, iou)
        # Compute 2 imgs, one with gt labels and other with detections labels
        Inference.subjective(
          stats=[(
            correct.cpu(),
            pred[:, 4].cpu(),
            pred[:, 5].cpu(),
            tcls
          )],
          detections=predn,
          labels=lcc,
          img=im,
          annotator=annotator,
          classnames=classnames
        )

      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)


  # compute stats
  results = Inference.compute_stats(stats)
  print(results)

cv2.destroyAllWindows()
