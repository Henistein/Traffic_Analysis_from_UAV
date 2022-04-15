import torch
import numpy as np
import cv2 
from models.yolo import Model
from multiprocessing import Process, Queue
from utils.conversions import xyxy2xywh
from utils.general import DetectionsMatrix
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

from inference import Inference, Annotator
from heatmap import HeatMap
from utils.conversions import scale_coords

def run_deepsort(model, video_path):
  inf = Inference(model=model, device='cuda', imsize=640, iou_thres=0.50, conf_thres=0.50)
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
  heatmap = HeatMap('image_registration/rua.png')
  annotator = Annotator()
  detections = DetectionsMatrix()
  #q = Queue()
  #p = Process(target=heatmap.draw_heatmap, args=(q,))
  #p.start()

  frame_id = 0 
  while cap.isOpened():
    ret, frame = cap.read()
    frame_id += 1
    if frame_id == 501: break

    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    # inference
    pred = inf.get_pred(frame)

    # Predictions in MOT format
    detections.update_current(
      bboxes=xyxy2xywh(pred[:, 0:4]),
      confs=pred[:, 4], # confs
      clss=pred[:, 5] # clss
    )

    # pass detections to deepsort
    outputs = deepsort.update(
      detections.current[:, 2:6], # xywhs
      detections.current[:, 6], # confs
      detections.current[:, 7], # clss
      frame.copy()
    )
    
    if len(outputs) > 0:
      # stack confs to outputs
      min_dim = min(outputs.shape[0], detections.current.shape[0])
      outputs = outputs[:min_dim]
      detections.current = detections.current[:min_dim]
      detections.current[:, 2:6] = outputs[:, :4] # bboxes
      detections.current[:, 1] = outputs[:, 4] + 1 # ids

      # add centers to heatmap
      heatmap.update_points(detections.current[:, 2:6])
      # update queue
      #q.put(heatmap.points_list)

      frame = inf.attach_detections(annotator, detections.current, frame, classnames, has_id=True)
      detections.update(append=False)
      # draw heatpoints in the frame
      frame = heatmap.draw_center(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break

  #p.join()
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  #run_visdrone()
  #run_coco()

  # run deepsort with yolo
  weights = 'weights/visdrone.pt'
  #weights = 'weights/yolov5l-xs.pt'

  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))
  run_deepsort(model, 'videos/rotunda.MP4')
