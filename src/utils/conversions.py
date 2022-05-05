import torch
import numpy as np

def xyxy2xywh(x):
  # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
  y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
  y[:, 2] = x[:, 2] - x[:, 0]  # width
  y[:, 3] = x[:, 3] - x[:, 1]  # height
  return y

def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def normyolo2xyxy(yolo_coords, w, h):
  # denormalize
  yolo_coords[:, [0, 2]] *= w
  yolo_coords[:, [1, 3]] *= h

  #x1 = yolo_coords[:, 0] - w/2
  #y1 = yolo_coords[:, 1] - h/2

  x1 = yolo_coords[:, 0] - yolo_coords[:, 2] / 2
  y1 = yolo_coords[:, 1] - yolo_coords[:, 3] / 2
  x2 = yolo_coords[:, 0] + yolo_coords[:, 2] / 2
  y2 = yolo_coords[:, 1] + yolo_coords[:, 3] / 2
  #x2 = x1 + yolo_coords[:, 2]
  #y2 = y1 + yolo_coords[:, 3]
  return torch.stack([x1, y1, x2, y2], dim=1)

def coco2xyxy(x):
  x = np.array(x)
  y = x.copy()
  y[:, 2] += x[:, 0]
  y[:, 3] += x[:, 1]
  return y

"""
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
"""

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # img1 (old) | img0 (new)
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

