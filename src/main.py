from lib2to3.pgen2.token import OP
import torch
import numpy as np
import cv2 
from tqdm import tqdm
from opts import OPTS
from utils.conversions import xyxy2xywh, xywh2xyxy
from utils.general import DetectionsMatrix, Annotator
from deep_sort.deep_sort import DeepSort

from scipy.spatial.distance import euclidean 
import matplotlib.pyplot as plt
from inference import Inference 
from heatmap import HeatMap
from utils.metrics import box_iou
from fastmcd.MCDWrapper import MCDWrapper
from counter import Box
from utils.map import MAP, Teste
from copy import deepcopy

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
      self.writer = cv2.VideoWriter('output.mp4', fourcc, self.fps, (self.width, self.height))

# auxiliar methods
def filter_current_ids(idcenters, current_ids):
  return {k:idcenters[k] for k in current_ids}


def run(model, opt):
  inf = Inference(model=model, device='cuda', imsize=opt.img_size, iou_thres=opt.iou_thres, conf_thres=opt.conf_thres)
  classnames = model.names
  # load deepsort
  deepsort = DeepSort('osnet_ibn_x1_0_MSMT17', inf.device, strongsort=True)

  # heatmap
  #heatmap = HeatMap('image_registration/map_rotunda.png')
  #mcd = MCDWrapper()
  
  # map
  mapp = MAP(
      lat_north=40.273668,
      long_west=-7.506679,
      lat_south=40.268085,
      long_east=-7.493513,
      lat_center=40.271018,
      long_center=-7.500335,
      image=cv2.imread('MAPA.jpg')
  )
  teste = Teste(mapp)

  annotator = Annotator()
  detections = DetectionsMatrix(
    classes_to_eval=model.names,
    classnames=model.names
  )
  video = Video(video_path=opt.path, start_from=opt.start_from, video_out=opt.video_out)

  isFirst = True
  frame_id = 0 
  idcenters = None # id:center
  map_img, img_crop = None, None
  pbar=tqdm(video.cap.isOpened(), total=video.video_frames)
  k = 0
  speeds = {}
  last_scaled_pts = None
  while pbar:
    ret, frame = video.cap.read()
    frame_id += 1
    annotator.add_image(frame)

    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    # Background Subtraction
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if isFirst:
      mcd.init(gray)
      isFirst = False
    else:
      mask = mcd.run(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """

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
    if not opt.just_detector:
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
        detections.current[:, 2:6] = outputs[:, :4] # bboxes xyxy
        detections.current[:, 1] = outputs[:, 4] + 1 # ids

        # add centers to heatmap
        #heatmap.update_points(detections.current[:, 2:6])
        # update queue
        #q.put(heatmap.points_list)

        # Filter just the moving objects
        if False:
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
          iou_matrix = box_iou(det_bboxes, bs_bboxes).numpy()
          detections.current = detections.current[iou_matrix.sum(axis=1)>0]
          if detections.current.shape[0] == 0: continue
          if len(detections.current.shape) == 1:
            detections.current = detections.current.reshape(1, -1) 


        # calculate centers of each bbox per id
        detections.update_idcenters()
        #of.update_centers(idcenters)
        #of.get_speed_ppixel(annotator)

      else:
        detections.current[:, 2:6] = xywh2xyxy(detections.current[:, 2:6])
    else:
      detections.current[:, 2:6] = xywh2xyxy(detections.current[:, 2:6])

    # MAP
    if frame_id % 3 == 0:
      map_img, img_crop, scaled_points = teste.get_next_data(k,filter_current_ids(detections.idcenters,detections.current[:, 1]))
      # speed
      if last_scaled_pts is not None:
        # calulate euclidean distance
        for id_ in set(scaled_points).intersection(set(last_scaled_pts)):
          dist = euclidean(last_scaled_pts[id_], scaled_points[id_])
          dist = dist*200/824 # convert pixels to meters (824px = 200m)
          if id_ in speeds.keys():
            speeds[id_].append((dist/0.1)*3.6)
          else:
            speeds[id_] = [(dist/0.1)*3.6]
      
      # update last_scaled_pts
      last_scaled_pts = deepcopy(scaled_points)
      k+=1

    # draw heatpoints in the frame
    #frame = heatmap.draw_heatpoints(frame)

    # update pbar
    pbar.update(1)

    # Draw counter
    #frame = cv2.rectangle(frame,(229,307),(231,411),(0,255,0),2)

    if opt.video_out:
      video.writer.write(frame)
    
    # draw 
    if not opt.no_show:
      # draw detections
      frame = inf.attach_detections(annotator, detections.current, classnames, label="I" if not opt.just_detector else "CP", speeds=speeds)
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
  video.cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  opt = OPTS.main_args()

  # run deepsort with yolo
  weights = 'weights/'+opt.model
  model = torch.load(weights)['model'].float()
  model.to(torch.device('cuda'))

  run(model, opt)
