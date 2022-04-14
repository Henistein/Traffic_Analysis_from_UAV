import torch
import numpy as np
import cv2 
from models.yolo import Model
from multiprocessing import Process, Queue
from utils.conversions import xyxy2xywh
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
  #q = Queue()
  #p = Process(target=heatmap.draw_heatmap, args=(q,))
  #p.start()

  frame_id = 0 
  mot = np.zeros((9, 10))
  while cap.isOpened():
    ret, frame = cap.read()
    frame_id += 1
    if frame_id == 501: break

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
      #heatmap.update_points(outputs[:, :4])
      # update queue
      #q.put(heatmap.points_list)

      confs = confs.unsqueeze(0).numpy().T
      outputs = np.append(np.append(outputs[:, :5], confs, axis=1), outputs[:, 5].reshape(-1, 1), axis=1) # hack to be [bboxes, id, conf, class]
      
      # output in mot format
      filter_outputs = outputs[outputs[:, -1] == 3]
      aux = np.full((len(filter_outputs),1), frame_id)
      aux = np.concatenate((aux, filter_outputs[:, 4].reshape(-1, 1)), axis=1)
      # xyxy2xywh
      filter_outputs[:, 2] -= filter_outputs[:, 0]
      filter_outputs[:, 3] -= filter_outputs[:, 1]
      aux = np.concatenate((aux, filter_outputs[:, :4]), axis=1)
      aux = np.concatenate((aux, confs[:len(filter_outputs)]), axis=1)
      aux = np.concatenate((aux, np.full((len(filter_outputs), 3), -1)), axis=1)
      mot = np.concatenate((mot, aux))
      

    #frame = inf.attach_detections(annotator, outputs, frame, classnames, has_id=True)
    # draw heatpoints in the frame
    #frame = heatmap.draw_center(frame)

    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
      break
  np.savetxt('mot500', mot, delimiter=', ', fmt='%f')
  print(mot)
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
  run_deepsort(model, 'videos/rotunda2.MP4')
