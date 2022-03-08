import torch
import numpy as np

def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y

def normyolo2xyxy(yolo_coords, w, h):
  # denormalize
  yolo_coords[:, [0, 2]] *= w
  yolo_coords[:, [1, 3]] *= h

  x1 = yolo_coords[:, 0] - yolo_coords[:, 2] / 2
  y1 = yolo_coords[:, 1] - yolo_coords[:, 3] / 2
  x2 = yolo_coords[:, 0] + yolo_coords[:, 2] / 2
  y2 = yolo_coords[:, 1] + yolo_coords[:, 3] / 2
  return torch.stack([x1, y1, x2, y2], dim=1)

def coco2xyxy(x):
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 2] += x[:, 0]
  y[:, 3] += x[:, 1]
  return y

def scale_coords(old_shape, new_shape, coords):
  h, w = old_shape
  newH, newW = new_shape

  gainH = newH / h
  gainW = newW / w
  pad = (newW - w * gainW) / 2, (newH - h * gainH) / 2  # wh padding

  coords[:, [0, 2]] -= pad[0]  # x padding
  coords[:, [1, 3]] -= pad[1]  # y padding

  coords[:, 0] /= gainW
  coords[:, 1] /= gainH
  coords[:, 2] /= gainW
  coords[:, 3] /= gainH

  # convert to homogeneous coordinate system
  coords = clip_coords((h, w), coords)
  coords = coords.detach().numpy()

  return coords

def clip_coords(shape, coords):
  h, w = shape
  coords[:, 0].clamp_(0, w)  # x1
  coords[:, 1].clamp_(0, h)  # y1
  coords[:, 2].clamp_(0, w)  # x2
  coords[:, 3].clamp_(0, h)  # y2

  return coords

