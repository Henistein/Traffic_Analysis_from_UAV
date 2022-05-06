import cv2
from scipy.spatial import distance
import numpy as np
import math
from skimage.transform import EssentialMatrixTransform
from skimage.measure import ransac

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
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  
  def update_centers(self, idcenters):
    self.objects_centers.insert(0, idcenters)
    self.objects_centers = self.objects_centers[:OpticalFlow.MAX_SIZE]
  
  
  def get_speed_ppixel(self, R, tvec, annotator=None):
    if len(self.objects_centers) >= 2:
      Rinv = np.linalg.inv(R)
      # get the same ids
      old_idc = self.objects_centers[1]
      new_idc = self.objects_centers[0]
      ids_intersection = [i for i in old_idc.keys() if i in new_idc.keys()]

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

      print(ids_speed[6] if 6 in ids_speed.keys() else None)

  def _is_inside_bbox(self, pt, bbox):
    a,b,c,d = bbox
    x,y = pt
    if (a < x < c) and (b < y < d):
      return True
    return False
  
  def normalize_kps(self, kps):
    return np.stack([np.dot(self.Kinv, np.array([pt[0], pt[1], 1]).T).T[0:2] for pt in kps])

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
    for bbox in bboxes:
      aux_func = lambda pt: not self._is_inside_bbox(pt, bbox)
      filt = list(map(aux_func, kps))
      kps = kps[filt]
      des = des[filt]

    return kps, des

  def poseRt(self, R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

  def fundamentalToRt(self, F):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float) 
    U, d, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
      U *= -1.0
    if np.linalg.det(Vt) < 0:
      Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
      R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

    # TODO: Resolve ambiguities in better ways. This is wrong.
    if t[2] < 0:
      t *= -1

    return R,t#np.linalg.inv(self.poseRt(R, t))
  
  def match_features(self, f1, f2):
    matches = self.bf.knnMatch(f1['des'], f2['des'], k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    idx1s, idx2s = set(), set()
    
    # Apply ratio test
    for m, n in matches:
      if m.distance < 0.75*n.distance:
        pt1 = f1['nkps'][m.queryIdx]
        pt2 = f2['nkps'][m.trainIdx]
        # be within orb distance 32                              
        if m.distance < 32:                                      
          # keep around indices                                  
          if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
            idx1.append(m.queryIdx)                              
            idx2.append(m.trainIdx)                              
            idx1s.add(m.queryIdx)                                
            idx2s.add(m.trainIdx)                                
            ret.append((pt1, pt2))
    
    # Filter using Fundamental Matrix
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                             EssentialMatrixTransform,
                             min_samples=8,
                             residual_threshold=0.02,
                             max_trials=100)

    return (idx1[inliers], idx2[inliers]), model.params
