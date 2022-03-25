import torch
import torch.nn as nn
import cv2
import numpy as np
from utils.general import non_max_suppression, letterbox, image_loader
from utils.metrics import ConfusionMatrix, process_batch, ap_per_class
from utils.conversions import coco2xyxy, scale_coords, normyolo2xyxy, clip_coords
from models.yolo import Model


def get_pred(model, img):
  '''
  returns prediction in numpy array
  '''
  model.eval()
  imsize = 640
  img, h, w = image_loader(img,imsize)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  img = img.to(device)
  pred = model(img)[0]

  pred = pred.cpu()

  pred = non_max_suppression(pred, conf_thres=0.45)[0] # conf_thres is confidence thresold

  if pred is not None:
    # scale coords to match true image size
    #pred = scale_coords((h, w), img[0].shape[1:], pred)
    #pred = scale_coords((h, w), pred, img[0].shape[1:])
    pred = scale_coords(img[0].shape[1:], pred, (h,w))
  
  return pred
                


def show_detections(detections, image, labels, ret=False, is_label=False):
  img = image.copy()
  for pred in detections:
    x1,y1,x2,y2 = map(int, pred[:4])

    c = pred[-1].item() # class
    p = pred[-2].item() # confidence (proability)

    start = (x1,y1)
    end = (x2,y2)

    if is_label:
      pred_data = f'{labels[c]}'
    else:
      pred_data = f'{labels[c]} {str(p*100)[:5]}%'

    color = (0,255,0)
    img = cv2.rectangle(img, start, end, color)
    img = cv2.putText(img, pred_data, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 

  # return img or show it
  if ret: 
    return img.copy()
  else:
    cv2.imshow('frame', img)
    cv2.waitKey(0)


def compute_stats(stats):

  # compute metrics
  stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
  
  if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=80)  # number of targets per class
    return mp, mr, map50, map
  else:
    nt = torch.zeros(1)
    return nt


def subjective(*stats, detections, labels, img):
  mp, mr, map50, map = compute_stats([stats])
  if map50 < 0.5:
    print(f"mAP50: {map50}")
    
    # get detections images and labels image with bb
    det_img = show_detections(detections, img, ret=True)
    l = torch.column_stack((labels[:, 1:], labels[:, 0]))
    lab_img = show_detections(l, img, ret=True, is_label=True)

    # concatenate det_img and lab_img vertical
    res = cv2.hconcat([det_img, lab_img])
    cv2.imshow('frame', res)
    cv2.waitKey(0)
