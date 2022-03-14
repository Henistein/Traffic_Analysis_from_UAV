import torch
import numpy as np
import cv2 
from models.yolo import Model

from inference import get_pred, show_detections


def run(model, video_path, classnames):
  cap = cv2.VideoCapture(video_path)
  while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

    # inference
    pred = get_pred(model, frame)

    if pred is not None:
      pred = torch.tensor(pred)
      # show image with bounding boxes
      frame = show_detections(pred, frame, classnames, ret=True)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

def run_coco():
  # load classnames
  classnames = [cl.strip() for cl in open('coco_classes.txt').readlines()]
  classnames = {classnames.index(k):k for k in classnames}
  cfg = 'cfg.yaml'
  video_path = '../traffic.mp4'
  weights = 'weights/yolov5x.pt'

  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))
  
  run(model, video_path, classnames)

def run_visdrone():
  # load classnames
  classnames = [cl.strip() for cl in open('visdrone_classes.txt').readlines()]
  classnames = {classnames.index(k):k for k in classnames}
  cfg = 'cfg.yaml'
  video_path = '../traffic.mp4'
  weights = 'weights/visdrone.pth'

  model = Model(cfg, ch=3, nc=8)
  model.load_state_dict(torch.load(weights))
  #model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))

  frame = cv2.imread('sample.jpg')
  frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

  # inference
  pred = get_pred(model, frame)

  #run(model, video_path, classnames)

if __name__ == '__main__':
  #run_visdrone()
  run_coco()
