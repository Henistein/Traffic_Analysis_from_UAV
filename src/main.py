from lib2to3.pgen2.token import OP
import torch
import numpy as np
import cv2 
from tqdm import tqdm
from opts import OPTS
from utils.conversions import xyxy2xywh, xywh2xyxy
from utils.general import DetectionsMatrix, Annotator
from deep_sort.deep_sort import DeepSort

from utils.optical_flow import OpticalFlow
from inference import Inference 
from heatmap import HeatMap
from utils.metrics import box_iou
from fastmcd.MCDWrapper import MCDWrapper
from counter import Box
from utils.map import MAP, Teste

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
        idcenters = detections.get_idcenters()
        #of.update_centers(idcenters)
        #of.get_speed_ppixel(annotator)

      else:
        detections.current[:, 2:6] = xywh2xyxy(detections.current[:, 2:6])
    else:
      detections.current[:, 2:6] = xywh2xyxy(detections.current[:, 2:6])

    # AQUI
    # homography sattelite
    if frame_id % 3 == 0:
      map_img, img_crop = teste.get_next_data(k)

      if frame_id < 480:
        src_pts = np.array([(592, 455), (596, 285), (592, 588), (744, 418), (745, 461), (793, 440), (383, 444)])
        dst_pts = np.array([(1713, 2446), (1736, 2326), (1721, 2550), (1826, 2431), (1826, 2469), (1875, 2450), (1590, 2434)])
      else:
        src_pts = np.array([(542, 403), (558, 304), (650, 566), (650, 313), (676, 105), (802, 206), (1145, 269)])
        dst_pts = np.array([(1672, 2013), (1687, 1951), (1743, 2137), (1751, 1963), (1773, 1820), (1856, 1897), (2070, 1951)])

      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      #dst = cv2.warpPerspective(frame, M, (mapp.image.shape[1], mapp.image.shape[0]))
      # resize to 1280 720
      #dst = cv2.resize(dst, (1280,720),cv2.INTER_AREA)
      #cv2.imshow('warp_frame', dst)

      scaled_centers = []
      if idcenters is not None:
        for pt in idcenters.values():
          scl_pt = [pt[0],pt[1],1]
          scl_pt = M@scl_pt
          scl_pt = [int(scl_pt[0]/scl_pt[2]), int(scl_pt[1]/scl_pt[2])]
          scl_pt = [int((scl_pt[0])/3.75), int((scl_pt[1])/3.865)]
          scaled_centers.append(scl_pt)
          # print scaled centers in small image
          map_img = cv2.circle(map_img, scl_pt, radius=3, color=(255,0,0), thickness=3)
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
      frame = inf.attach_detections(annotator, detections.current, classnames, has_id=True if not opt.just_detector else False)
      # draw centers
      if idcenters is not None: annotator.draw_centers(idcenters.values())
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
