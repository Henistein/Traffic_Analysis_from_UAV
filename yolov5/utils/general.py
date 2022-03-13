import torch
import time
import torchvision
import numpy as np
import cv2
from utils.conversions import xywh2xyxy

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, fast=False, classes=None, agnostic=False):
  """Performs Non-Maximum Suppression (NMS) on inference results
  Returns:
       detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
  """
  if prediction.dtype is torch.float16:
      prediction = prediction.float()  # to FP32

  nc = prediction.shape[2] - 5  # number of classes
  nc = 8

  xc = prediction[..., 4] > conf_thres  # candidates

  # Settings
  min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
  max_det = 300  # maximum number of detections per image
  time_limit = 10.0  # seconds to quit after
  redundant = True  # require redundant detections
  fast |= conf_thres > 0.001  # fast mode
  multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

  if fast:
      merge = False
  else:
      merge = True  # merge for best mAP (adds 0.5ms/img)

  t = time.time()
  output = [None] * prediction.shape[0]
  for xi, x in enumerate(prediction):  # image index, image inference
      # Apply constraints
      #x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
      x = x[xc[xi]]  # confidence

      # If none remain process next image
      if not x.shape[0]:
        continue

      # Compute conf
      x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

      # Box (center x, center y, width, height) to (x1, y1, x2, y2)
      box = xywh2xyxy(x[:, :4])

      # Detections matrix nx6 (xyxy, conf, cls)
      if multi_label:
          i, j = (x[:, 5:] > conf_thres).nonzero().t()
          x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
      else:  # best class only
          conf, j = x[:, 5:].max(1, keepdim=True)
          x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

      # Filter by class
      if classes:
          x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

      # Apply finite constraint
      # if not torch.isfinite(x).all():
      #     x = x[torch.isfinite(x).all(1)]

      # If none remain process next image
      n = x.shape[0]  # number of boxes
      if not n:
          continue

      # Sort by confidence
      # x = x[x[:, 4].argsort(descending=True)]

      # Batched NMS
      c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
      boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
      i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
      if i.shape[0] > max_det:  # limit detections
          i = i[:max_det]
      if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
          try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
              iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
              weights = iou * scores[None]  # box weights
              x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
              if redundant:
                  i = i[iou.sum(1) > 1]  # require redundancy
          except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
              print(x, i, x.shape, i.shape)
              pass

      output[xi] = x[i]
      if (time.time() - t) > time_limit:
          break  # time limit exceeded

  return output


def box_iou(box1, box2):
  # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
  """
  Return intersection-over-union (Jaccard index) of boxes.
  Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
  Arguments:
      box1 (Tensor[N, 4])
      box2 (Tensor[M, 4])
  Returns:
      iou (Tensor[N, M]): the NxM matrix containing the pairwise
          IoU values for every element in boxes1 and boxes2
  """

  def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])

  area1 = box_area(box1.T)
  area2 = box_area(box2.T)

  # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
  inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
  return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
  # Resize and pad image while meeting stride-multiple constraints
  shape = im.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better val mAP)
      r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
      dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
  elif scaleFill:  # stretch
      dw, dh = 0.0, 0.0
      new_unpad = (new_shape[1], new_shape[0])
      ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
      im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  return im, ratio, (dw, dh)

def image_loader(im,imsize):
  '''
  processes input image for inference 
  '''
  h, w = im.shape[:2]
  im = letterbox(im, (640, 640), stride=32)[0]
  im = im.transpose((2, 0, 1))[::-1]
  im = np.ascontiguousarray(im)
  im = torch.from_numpy(im)
  im = im.float()
  im /= 255.0
  im = im.unsqueeze(0)

  return im, h, w 
