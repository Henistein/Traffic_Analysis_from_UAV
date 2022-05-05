import cv2
from scipy.spatial import distance
import numpy as np
import math

class OpticalFlow:
  MAX_SIZE = 20
  def __init__(self):
    self.objects_centers = []
    # camera calibration
    self.W = 1280
    self.H = 720
    self.fov_angle = 84*math.pi/180
    self.F = int((self.W/2) / math.tan(self.fov_angle/2))
    self.K = np.array([[self.F,0,self.W//2],[0,self.F,self.H//2],[0,0,1]])
    self.Kinv = np.linalg.inv(self.K)
    self.orb = cv2.ORB_create()
  
  def update_centers(self, idcenters):
    self.objects_centers.insert(0, idcenters)
    self.objects_centers = self.objects_centers[:OpticalFlow.MAX_SIZE]
  
  
  def get_speed_ppixel(self, annotator=None):
    if len(self.objects_centers) >= 2:
      # get the same ids
      old_idc = self.objects_centers[1]
      new_idc = self.objects_centers[0]
      ids_intersection = [i for i in old_idc.keys() if i in new_idc.keys()]

      R = np.eye(3)
      Rinv = np.linalg.inv(R)
      tvec = np.array([0,0,32])

      # calculate the speed
      ids_speed = {}
      for _id in ids_intersection:
        # normalize points with intrinsics
        nold_pt = np.dot(self.Kinv, np.array([old_idc[_id][0], old_idc[_id][1], 1]))
        nnew_pt = np.dot(self.Kinv, np.array([new_idc[_id][0], new_idc[_id][1], 1]))
        
        nold_pt = Rinv.dot(nold_pt - tvec)
        nnew_pt = Rinv.dot(nnew_pt - tvec)

        # euclidian distance between centers
        ids_speed[_id] = distance.euclidean(nold_pt, nnew_pt) / (1/30)
        # draw trail (optional)
        pair_points = [list(it[_id]) for it in self.objects_centers if _id in it.keys()] # get old pair points
        if annotator: annotator.draw_trail(pair_points)

      print(ids_speed[2] if 2 in ids_speed.keys() else None)

  def _is_inside_bbox(self, pt, bbox):
    a,b,c,d = bbox
    x,y = pt
    if (a < x < c) and (b < y < d):
      return True
    return False

  def _extract_keypoints(self, img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # detection
    pts = cv2.goodFeaturesToTrack(gray, 3000, qualityLevel=0.01, minDistance=7)
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = self.orb.compute(img, kps)

    # return pts and des
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

  def extract_features(self, img, bboxes):
    """
    extract features (keypoints) and remove those who are inside detections
    """
    kps, des = self._extract_keypoints(img)
    filtered_kps = []
    for bbox in bboxes:
      aux_func = lambda pt: not self._is_inside_bbox(pt, bbox)
      kps = kps[list(map(aux_func, kps))]

    return kps