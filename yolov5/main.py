import torch
import numpy as np
import cv2 
from tqdm import tqdm
from models.yolo import Model
from multiprocessing import Process, Queue
from utils.conversions import xyxy2xywh, xywh2xyxy
from utils.general import DetectionsMatrix
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

from inference import Inference, Annotator
from heatmap import HeatMap
from utils.conversions import scale_coords
from utils.metrics import box_iou
from fastmcd.MCDWrapper import MCDWrapper
from counter import Box

def run_deepsort(model, video_path):
  inf = Inference(model=model, device='cuda', imsize=1920, iou_thres=0.50, conf_thres=0.50)
  classnames = model.names
  cap = cv2.VideoCapture(video_path)
  # load deepsort
  deepsort = DeepSort('osnet_ibn_x1_0_MSMT17', inf.device, strongsort=True)

  # heatmap
  #heatmap = HeatMap('image_registration/map_rotunda.png')
  annotator = Annotator()
  detections = DetectionsMatrix()
  mcd = MCDWrapper()
  box_counter = Box((229, 307), (231, 411))
  #q = Queue()
  #p = Process(target=heatmap.draw_heatmap, args=(q,))
  #p.start()
  video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  time_length = int(video_frames / fps)
  #cap.set(1, (1600 /(time_length*fps)))
  #cap.set(cv2.CAP_PROP_POS_MSEC, 56000)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  video = cv2.VideoWriter('output.mp4', fourcc, fps, (1280, 720))

  isFirst = True
  frame_id = 0 
  pbar=tqdm(cap.isOpened(), total=video_frames)
  while pbar:
    ret, frame = cap.read()
    frame_id += 1


    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    # Background Subtraction
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if isFirst:
      mcd.init(gray)
      isFirst = False
    else:
      mask = mcd.run(gray)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # inference
    pred = inf.get_pred(frame)
    if pred is None: continue

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
      #heatmap.update_points(detections.current[:, 2:6])
      # update queue
      #q.put(heatmap.points_list)

      # Filter just the moving objects
      if True:
        frame[mask > 0, 2] = 255
        bs_bboxes = []
        for cnt in contours:
          area = cv2.contourArea(cnt)
          # threshold
          if area > 0:
            x,y,w,h = cv2.boundingRect(cnt)
            bs_bboxes.append([x,y,w,h])

        if len(bs_bboxes) == 0: continue
        bs_bboxes = xywh2xyxy(torch.tensor(bs_bboxes))
        det_bboxes = torch.tensor(detections.current[:, 2:6])
        iou_matrix = box_iou(det_bboxes, bs_bboxes)
        detections.current = detections.current[iou_matrix.sum(axis=1)>0]
        if len(detections.current.shape) == 1:
          detections.current = detections.current.reshape(1, -1) 

      for det_boxes in detections.current[:, 2:6]:
        if box_counter.overlap(det_boxes[:2], det_boxes[2:]):
                box_counter.frame_countdown -= 1
                if box_counter.frame_countdown <= 0:
                    box_counter.counter += 1
                    print(box_counter.counter)
                box_counter.frame_countdown = 10
        
      frame = inf.attach_detections(annotator, detections.current, frame, classnames, has_id=True)
      detections.update(append=False)
      # draw heatpoints in the frame
      #frame = heatmap.draw_heatpoints(frame)

      pbar.update(1)

    # Draw counter
    frame = cv2.rectangle(frame,(229,307),(231,411),(0,255,0),2)

    video.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break

  #p.join()
  video.release()
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  #run_visdrone()
  #run_coco()

  # run deepsort with yolo
  #weights = 'weights/visdrone.pt'
  weights = 'weights/yolov5l-xs.pt'

  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))
  run_deepsort(model, 'videos/rotunda2.MP4')
  #run_deepsort(model, 'videos/non_stationary.mp4')
