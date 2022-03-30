import cv2
import numpy as np

pts1 = np.array([(669, 639), (472, 586), (709, 428), (761, 395), (879, 313), (953, 264), (334, 687), (644, 473), (804, 365), (909, 297), (615, 562), (671, 519), (721, 479), (757, 451), (794, 423)])
pts2 = np.array([(104, 46), (165, 53), (164, 134), (162, 162), (156, 271), (156, 389), (163, 23), (163, 102), (158, 196), (156, 305), (140, 69), (139, 87), (137, 105), (137, 128), (137, 146)])

class HeatMap:
  def __init__(self, map_img_path):
    self.heat_points = {}
    self.map_img = cv2.imread(map_img_path, 0)
    #self.map_img = self.map_img.reshape(self.map_img.shape+(1,))
    self.mask = np.zeros(self.map_img.shape+(3,))
    self.heatmap = None

  def numpy_points(self):
    ret= np.array(list(self.heat_points.keys()))
    return ret

  def update_points(self, bboxes):
    """
    Extract centers from the bboxes and update the points
    """
    for det in bboxes:
      x1,y1,x2,y2 = det
      center = (int((x1+x2)/2), int((y1+y2)/2))
      if center in self.heat_points.keys():
        self.heat_points[center] += 1
      else:
        self.heat_points[center] = 1

  def update_heatmap(self):
    color = ()
    for point in self.heat_points:
      occ = self.heat_points[point]

      if px > self.mask.shape[1] or py > self.mask.shape[0]:
        continue
      occ *= 10
      if occ > 765:
        color = (255, 255, 255)
      elif occ > 510:
        color = (255, 255, occ%510)
      elif occ > 255:
        color = (255, occ%255, 0)
      else:
        color = (occ, 0, 0)
      self.mask[(px, py)] = list(color) 
      print(self.mask[(px, py)])

  def calc_homograpy_matrix(self, pts1, pts2):
    # find homography
    self.M, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

  def apply_matrix(self, point):
    # apply matrix to point
    pt = np.array((point[0], point[1], 1)).reshape(3, 1) 
    px, py, pz = M.dot(pt)
    # convert homogeneous to cartesian
    px = np.int(np.round(px/pz))
    py = np.int(np.round(py/pz))
    return px, py
      
  def draw_center(self, frame):
    """
    Draw heat points on frame
    """
    for p in self.heat_points:
      frame = cv2.circle(frame, p, radius=2, color=(0, 0, 255), thickness=-1)
    return frame
