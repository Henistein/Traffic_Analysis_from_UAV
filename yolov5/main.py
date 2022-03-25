import torch
import numpy as np
import cv2 
from models.yolo import Model
from utils.conversions import xyxy2xywh
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

from inference import get_pred, show_detections
from utils.conversions import scale_coords


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
      #pred = pred.clone().detach().cpu()
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
  weights = 'weights/visdrone100.pt'

  #model = Model(cfg, ch=3, nc=10)
  #model.load_state_dict(torch.load(weights))
  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))

  run(model, video_path, classnames)

def run_deepsort(model, video_path):
  model.eval()
  classnames = {model.names.index(k):k for k in model.names}
  cap = cv2.VideoCapture(video_path)
  # load deepsort
  cfg = get_config()
  cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
  deepsort = DeepSort('osnet_ibn_x1_0_MSMT17',
                      torch.device('cuda'),
                      max_dist=cfg.DEEPSORT.MAX_DIST,
                      max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                      )
  while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    orig = frame.copy()
    #h,w = frame.shape[:2]
    #print(h,w)
    #frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
    img = frame.copy()

    # inference
    pred = get_pred(model, frame)

    xywhs = xyxy2xywh(pred[:, 0:4])
    confs = pred[:, 4]
    clss = pred[:, 5]

    # pass detections to deepsort
    outputs = deepsort.update(
      xywhs.cpu(),
      confs.cpu(),
      clss.cpu(),
      frame.copy()
    )
    if len(outputs) > 0:
      for j, (output, conf) in enumerate(zip(outputs, confs)):
        bboxes = output[0:4]
        id = output[4]
        cls = output[5]

        c = int(cls) # integer class
        label = f'{id} {classnames[c]} {conf:.2f}'

        # scale  bboxes to original
        #bboxes = scale_coords((640, 640), np.expand_dims(bboxes, 0).astype(np.float64), (h,w))[0].astype(np.int64)

        start = bboxes[[0, 1]]
        end = bboxes[[2, 3]]


        color = (0,255,0)
        img = cv2.rectangle(img, start, end, color)
        img = cv2.putText(img, label, (bboxes[0],bboxes[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 

        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  #run_visdrone()
  #run_coco()

  # run deepsort with yolo
  weights = 'weights/visdrone100.pt'

  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))
  run_deepsort(model, 'videos/drone1.MP4')
