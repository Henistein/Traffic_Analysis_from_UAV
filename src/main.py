from distutils.command.config import config
import torch
import numpy as np
import cv2 
from tqdm import tqdm
from opts import OPTS
from utils.conversions import xyxy2xywh, xywh2xyxy
from utils.general import DetectionsMatrix, Annotator
from deep_sort.deep_sort import DeepSort
from utils.speed_handler import SpeedHandler
from utils.counter import Counter 
import glob

from utils.inference import Inference 
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
  # check if yolo or fasterrcnn
  is_yolo = True if opt.model.startswith('yolo') else False
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  inf = Inference(model=model, imsize=opt.img_size, iou_thres=opt.iou_thres, conf_thres=opt.conf_thres)

  # process input
  video_file,logs_file,mapp_file = process_input_folder(opt.path)
  if mapp_file is not None:
    # load deepsort
    deepsort = DeepSort('osnet_x0_25', device, 0.2, 0.7, 30, 3, 100)

  # annotator
  annotator = Annotator()
  # detections
  detections = DetectionsMatrix(
    classes_to_eval=model.names,
    classnames=model.names
  )
  # speed handler
  speed_handler = SpeedHandler()
  # video
  video = Video(video_path=video_file, start_from=opt.start_from, video_out=opt.video_out)
  
  if mapp_file is not None:
    # drone map
    geo = GeoRef(mapp_file)
    drone_map = MapDrone(logs_file,geo,video.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # counter
    counter = Counter()
    counter.update_img(geo.image, (geo.image.shape[1]/1280, geo.image.shape[0]/720))

  frame_id = -1
  map_img, img_crop = None, None
  pbar=tqdm(video.cap.isOpened(), total=video.video_frames)
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
    if is_yolo:
      pred = inf.get_pred(frame)
    else:
      pred = model.run_on_opencv_image(frame)

    if pred is None:
      # update pbar
      pbar.update(1)
      continue

    # Predictions in MOT format
    if is_yolo:
      detections.update_current(
        bboxes=xyxy2xywh(pred[:, 0:4]),
        confs=pred[:, 4], # confs
        clss=pred[:, 5] # clss
      )
    else:
      detections.update_current(
        bboxes=xyxy2xywh(pred.bbox),
        confs=pred.get_field('scores'),
        clss=pred.get_field('labels')-2
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

    # Drone MAP
    if frame_id % 3 == 0 and mapp_file is not None and (frame_id//3)<drone_map.max_data:
      map_img, img_crop, scaled_points = drone_map.get_next_data(
          frame_id//3,filter_current_ids(detections.idcenters,detections.current[:, 1])
      )

      if last_scaled_pts is not None:
        # calulate speed
        speed_handler.update_speeds(scaled_points["geo"], last_scaled_pts["geo"])

        # count cars
        counter.update_img(map_img, (drone_map.geo.image.shape[1]/1280, drone_map.geo.image.shape[0]/720))
        counter.count(scaled_points["px"])
      
      # update last_scaled_pts
      last_scaled_pts = deepcopy(scaled_points)


    # update pbar
    pbar.update(1)

    # draw detections
    frame = inf.attach_detections(annotator, detections.current, model.names, label="I" if not opt.just_detector else "CP", speeds=speed_handler.speeds)
    if opt.video_out:
      video.writer.write(frame)
    
    # draw 
    if not opt.no_show:
      # draw centers
      if len(detections.idcenters): annotator.draw_centers(filter_current_ids(detections.idcenters, detections.current[:, 1]).values())
      if counter.img is not None: cv2.imshow('map_img', counter.img)
      if img_crop is not None: cv2.imshow('crop_img', img_crop)

      cv2.imshow('frame', frame)
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

  weights = 'weights/'+opt.model

  print(f"Running {opt.model}")
  if opt.model.startswith('yolo'):
    # load yolo model
    model = torch.load(weights)['model'].float()
    model.to(torch.device('cuda')).eval()
  else:
    from maskrcnn_benchmark.config import cfg
    from utils.predictor import COCODemo
    # load fasterrcnn model
    config_file = "configs/fasterrcnn.yaml"
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weights])
    model = COCODemo(
      cfg,
      min_image_size=800,
      confidence_threshold=0.5
    )
    model.names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

  run(model, opt)
