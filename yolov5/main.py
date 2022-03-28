import torch
import numpy as np
import cv2 
from models.yolo import Model
from utils.conversions import xyxy2xywh
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

from inference import Inference
from heatmap import HeatMap
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
  weights = 'weights/visdrone.pt'

  #model = Model(cfg, ch=3, nc=10)
  #model.load_state_dict(torch.load(weights))
  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))

  run(model, video_path, classnames)

def run_deepsort(model, video_path):
  inf = Inference(model=model, device='cuda')
  classnames = model.names
  cap = cv2.VideoCapture(video_path)
  # load deepsort
  cfg = get_config()
  cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
  deepsort = DeepSort('osnet_ibn_x1_0_MSMT17',
                      inf.device,
                      max_dist=cfg.DEEPSORT.MAX_DIST,
                      max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                      )
  # heatmap
  heatmap = HeatMap()
  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    # inference
    pred = inf.get_pred(frame)

    xywhs = xyxy2xywh(pred[:, 0:4]).cpu()
    confs = pred[:, 4].cpu()
    clss = pred[:, 5].cpu()

    # pass detections to deepsort
    outputs = deepsort.update(
      xywhs,
      confs,
      clss,
      frame.copy()
    )
    
    if len(outputs) > 0:
      # stack confs to outputs
      min_dim = min(outputs.shape[0], confs.shape[0])
      outputs = outputs[:min_dim]
      confs = confs[:min_dim]

      # add centers to heatmap
      heatmap.update_points(outputs[:, :4])

      confs = confs.unsqueeze(0).numpy().T
      outputs = np.append(np.append(outputs[:, :5], confs, axis=1), outputs[:, 5].reshape(-1, 1), axis=1) # hack to be [bboxes, id, conf, class]
      

    frame = inf.attach_detections(inf, outputs, frame, classnames, has_id=True)
    # draw heatpoints in the frame
    frame = heatmap.draw_center(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  #run_visdrone()
  #run_coco()

  # run deepsort with yolo
  weights = 'weights/visdrone.pt'

  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))
  run_deepsort(model, 'videos/visdrone1.MP4')
