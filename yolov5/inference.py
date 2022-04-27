import torch
import cv2
import numpy as np
from utils.general import non_max_suppression, letterbox, image_loader
from utils.metrics import process_batch, ap_per_class
from utils.conversions import coco2xyxy, scale_coords, normyolo2xyxy, clip_coords
from models.yolo import Model


class Inference:
  def __init__(self, model, conf_thres=0.5, iou_thres=0.5, imsize=(640, 640), device=None):
    if not device:
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = torch.device(device)
    self.model = model.to(self.device).eval()
    self.annotator = Annotator()

    self.imsize = imsize
    self.conf_thres = conf_thres
    self.iou_thres = iou_thres


  def get_pred(self, img, scale=True):
    '''
    returns prediction in numpy array
    '''
    img, h, w = image_loader(img,self.imsize)
    img = img.to(self.device)

    pred = self.model(img)[0]
    pred = pred.cpu()
    pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0] # conf_thres is confidence thresold

    if pred is not None and scale:
      pred = scale_coords(img[0].shape[1:], pred, (h,w))
    
    if not scale: # return (h, w) and pred
      return (h, w), pred
    return pred
  
                
  @staticmethod
  def attach_detections(annotator, detections, img, classnames, has_id=False, is_label=False):
    """
    detections format: [N][frame_id, id, x, y, x, y, confs, cls]
    """
    img = img.copy()
    # add image to annotator
    annotator.add_image(img)
    for pred in detections:
      x1,y1,x2,y2 = map(int, pred[2:6])

      c = int(pred[-1].item()) # class
      p = pred[-2].item() # confidence (proability)
      id = int(pred[1])

      start = (x1,y1)
      end = (x2,y2)

      if has_id:
        label = f'{id}'
      elif is_label:
        label = f'{classnames[c]}' 
      else:
        label = f'{classnames[c]} {str(p*100)[:5]}%'

      # annotate image
      annotator.draw(start, end, label, c)

    return annotator.image


  @staticmethod
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


  @staticmethod
  def subjective(stats, threshold=0.50, **kwargs):
    """
    stats: stats
    kwargs: detections, scaled labels x1y1x2y2, img, annotator, classnames
    Note:
      detections format: [frame_id, id, x, y, x, y, confs, cls]
    """
    detections, labels, img, annotator, classnames = \
    kwargs['detections'], kwargs['labels'], kwargs['img'], kwargs['annotator'], kwargs['classnames']

    mp, mr, map50, map = Inference.compute_stats(stats)
    if map50 < threshold:
      print(f"mAP50: {map50}")
      
      # get detections images and labels image with bb
      det_img = Inference.attach_detections(annotator, detections, img, classnames, has_id=True)
      lab_img = Inference.attach_detections(annotator, labels, img, classnames, has_id=True)

      # concatenate det_img and lab_img vertical
      res = cv2.hconcat([det_img, lab_img])
      cv2.imshow('frame', res)
      cv2.waitKey(0)

class Annotator:
  def __init__(self):
    colors = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB')
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    self.palette = [hex2rgb('#' + c) for c in colors]
    self.img = None
  
  @property
  def image(self):
    return self.img.copy()

  def add_image(self, image):
    self.img = image
    self.lw =  max(round(sum(image.shape) / 2 * 0.003), 2)
  
  def _generate_color(self, i):
    # generate color by class
    return self.palette[int(i) % len(self.palette)]

  def draw(self, start, end, label, cls):
    color = self._generate_color(cls)
    self.img = cv2.rectangle(self.img, start, end, color, thickness=self.lw, lineType=cv2.LINE_AA)
    self.img = cv2.putText(self.img, label, (start[0], start[1]+25), 0, 0.5, color, thickness=max(self.lw - 1, 1), lineType=cv2.LINE_AA)
