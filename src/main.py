import torch
import numpy as np
import cv2 
from tqdm import tqdm
from opts import OPTS
from utils.conversions import xyxy2xywh, xywh2xyxy
from utils.general import DetectionsMatrix, Annotator
from deep_sort.deep_sort import DeepSort
import geopy.distance
import glob

from inference import Inference 
from counter import Box
from utils.dronemap import *
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

class Video:
  def __init__(self, video_path, start_from=None, video_out=False):
    self.cap = cv2.VideoCapture(video_path)
    self.video_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
    self.time_length = int(self.video_frames/self.fps)
    self.width = int(self.cap.get(cv2. CAP_PROP_FRAME_WIDTH))
    self.height = int(self.cap.get(cv2. CAP_PROP_FRAME_HEIGHT))
    self.video_out = video_out

    if start_from:
      self.cap.set(cv2.CAP_PROP_POS_MSEC, start_from)
    if video_out:
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      self.writer = cv2.VideoWriter('output.avi', fourcc, self.fps, (self.width, self.height))

# auxiliar methods
def filter_current_ids(idcenters, current_ids):
  return {k:idcenters[k]["last"] for k in current_ids if k != -1}

def process_input_folder(path):
  """
  input folder must contain:
  /logs  -> logs.csv
  /map   -> map.tif
  /video -> video.mp4
  if logs or map not available it just process video
  """
  logs = glob.glob(path+'/logs/*')
  mapp = glob.glob(path+'/map/*')
  video = glob.glob(path+'/video/*')

  if (len(logs) and len(mapp)) > 0:
    logs_file = logs[0]
    mapp_file = mapp[0]
  else:
    logs_file = None
    mapp_file = None

  if len(video) == 0:
    raise Exception("Video not found inside input folder")
  video_file = video[0]

  return video_file,logs_file,mapp_file

def run(model, opt):
  inf = Inference(model=model, device='cuda', imsize=opt.img_size, iou_thres=opt.iou_thres, conf_thres=opt.conf_thres)
  classnames = model.names

  # process input
  video_file,logs_file,mapp_file = process_input_folder(opt.path)
  if mapp_file is not None:
    # load deepsort
    deepsort = DeepSort('osnet_x0_25', inf.device, 0.2, 0.7, 30, 3, 100)

  # map
  annotator = Annotator()
  detections = DetectionsMatrix(
    classes_to_eval=model.names,
    classnames=model.names
  )

  # video
  video = Video(video_path=video_file, start_from=opt.start_from, video_out=opt.video_out)
  
  if mapp_file is not None:
    # drone map
    geo = GeoRef(mapp_file)
    drone_map = MapDrone(logs_file,geo,video.cap.get(cv2.CAP_PROP_FRAME_COUNT))

  frame_id = -1
  map_img, img_crop = None, None
  pbar=tqdm(video.cap.isOpened(), total=video.video_frames)
  speeds = {}
  last_scaled_pts = None
  while pbar:
    ret, frame = video.cap.read()
    frame_id += 1

    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    annotator.add_image(frame)

    if opt.n_frames == frame_id:
      break

    # inference
    pred = inf.get_pred(frame)
    if pred is None:
      # update pbar
      pbar.update(1)
      continue

    # Predictions in MOT format
    detections.update_current(
      bboxes=xyxy2xywh(pred[:, 0:4]),
      confs=pred[:, 4], # confs
      clss=pred[:, 5] # clss
    )
    # pass detections to deepsort
    if not opt.just_detector:
      outputs = deepsort.update(
        torch.tensor(detections.current[:, 2:6]), # xywhs
        torch.tensor(detections.current[:, 6]), # confs
        torch.tensor(detections.current[:, 7]), # clss
        frame.copy()
      )
    
      if len(outputs) > 0:
        # stack confs to outputs
        min_dim = min(outputs.shape[0], detections.current.shape[0])
        outputs = outputs[:min_dim]
        detections.current = detections.current[:min_dim]
        detections.current[:, 2:6] = outputs[:, :4] # bboxes xyxy
        detections.current[:, 1] = outputs[:, 4] + 1 # ids

        # calculate centers of each bbox per id
        detections.update_idcenters()

      else:
        detections.current[:, 2:6] = xywh2xyxy(detections.current[:, 2:6])
    else:
      detections.current[:, 2:6] = xywh2xyxy(detections.current[:, 2:6])

    # MAP
    if frame_id % 3 == 0 and mapp_file is not None and (frame_id//3)<drone_map.max_data:
      map_img, img_crop, scaled_points = drone_map.get_next_data(
          frame_id//3,filter_current_ids(detections.idcenters,detections.current[:, 1]),
          frame, detections.current[:, 2:6],
      )
      # calculate speed
      if last_scaled_pts is not None:
        # calulate euclidean distance
        for id_ in set(scaled_points).intersection(set(last_scaled_pts)):
          dist = geopy.distance.geodesic(last_scaled_pts[id_], scaled_points[id_]).meters
          if id_ in speeds.keys():
            speeds[id_].append((dist/0.1)*3.6)
          else:
            speeds[id_] = [(dist/0.1)*3.6]
      
      # update last_scaled_pts
      last_scaled_pts = deepcopy(scaled_points)

    # update pbar
    pbar.update(1)

    # draw detections
    frame = inf.attach_detections(annotator, detections.current, classnames, label="I" if not opt.just_detector else "CP", speeds=speeds)
    if opt.video_out:
      video.writer.write(annotator.image)
    
    # draw 
    if not opt.no_show:
      # draw centers
      if len(detections.idcenters): annotator.draw_centers(filter_current_ids(detections.idcenters, detections.current[:, 1]).values())
      if map_img is not None: cv2.imshow('map_img', map_img)
      if img_crop is not None: cv2.imshow('crop_img', img_crop)

      cv2.imshow('frame', annotator.image)
      if cv2.waitKey(1) == ord('q'):
        break
    
    # update detections
    detections.update(append=True if opt.labels_out else False)

  if opt.labels_out:
    # save labels
    detections.mot_matrix[:, 2:6] = xywh2xyxy(detections.mot_matrix[:, 2:6])
    np.savetxt('outputs.txt', detections.mot_matrix, delimiter=',')

  if opt.video_out:
    # save video
    video.writer.release()
  
  # show heatmap
  drone_map.heatmap.draw_heatmap()

  
  video.cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  opt = OPTS.main_args()

  # run deepsort with yolo
  weights = 'weights/'+opt.model
  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))

  run(model, opt)
