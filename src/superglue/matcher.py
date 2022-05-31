import cv2
import torch

from deep_sort.sort import detection


from .supermodels.matching import Matching
from .supermodels.utils import *

torch.set_grad_enabled(False)

class Matcher:
  RESIZE = [1280]
  MAX_KEYPOINTS = 2048
  KEYPOINT_THRESHOLD = 0.005
  NMS_RADIUS = 3
  SINKHORN_ITERATIONS = 20
  MATCH_THRESHOLD = 0.2
  SUPERGLUE = "outdoor"

  def __init__(self):
    self.device = 'cuda'
    self.config = {
        'superpoint': {
            'nms_radius': Matcher.NMS_RADIUS,
            'keypoint_threshold': Matcher.KEYPOINT_THRESHOLD,
            'max_keypoints': Matcher.MAX_KEYPOINTS
        },
        'superglue': {
            'weights': Matcher.SUPERGLUE,
            'sinkhorn_iterations': Matcher.SINKHORN_ITERATIONS,
            'match_threshold': Matcher.MATCH_THRESHOLD,
        }
    }
    # Load the SuperPoint and SuperGlue models.
    self.model = Matching(self.config).eval().to(self.device)

  def load_image_pair(self, frame0, frame1):
    # image pair
    self.image0, self.inp0, self.scales0 = read_image(frame0, self.device, [frame0.shape[1], frame0.shape[0]], 0, True)
    self.image1, self.inp1, self.scales1 = read_image(frame1, self.device, [frame1.shape[1], frame1.shape[0]], 0, True)

  def matching(self):
    # Perform the matching.
    pred = self.model({'image0': self.inp0, 'image1': self.inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    return mkpts0, mkpts1
  
  def filter_mkps(self, mkpts1, mkpts2, detections):
    new_mkpts1 = []
    new_mkpts2 = []
    # if point is not inside any bbox, add it to new_mkps
    for i in range(len(mkpts1)):
      inside = False
      for bbox in detections:
        # check if point is inside bbox
        if bbox[0] < mkpts1[i][0] < bbox[2] and bbox[1] < mkpts1[i][1] < bbox[3]:
          inside = True
          break
      if not inside:
        new_mkpts1.append([mkpts1[i][0], mkpts1[i][1]])
        new_mkpts2.append([mkpts2[i][0], mkpts2[i][1]])

    return np.array(new_mkpts1), np.array(new_mkpts2)

  
  def get_warped_image(self, frame0, frame1, detections):
    # load image pair
    self.load_image_pair(frame0, frame1)
    # match frames
    mkpts0, mkpts1 = self.matching()
    mkpts0, mkpts1 = self.filter_mkps(mkpts0, mkpts1, detections)
    # find homography
    H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    res = cv2.warpPerspective(self.image0, H, (self.image1.shape[1], self.image1.shape[0]))
    return H, res


if __name__ == '__main__':

  vid1 = cv2.VideoCapture("videos/moving_drone.mp4")
  vid2 = cv2.VideoCapture("videos/output.mp4")

  matcher = Matcher()

  i = 0
  while True:
    ret0, frame0 = vid1.read()
    ret1, frame1 = vid2.read()

    res = matcher.get_warped_image(frame0, frame1)

    cv2.imwrite("frame%05d.png"%i, res)

    if not ret0 or not ret1:
        break
    i += 1 
    
