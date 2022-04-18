import cv2
import numpy as np
import io
from scipy.stats.kde import gaussian_kde
from PIL import Image
import matplotlib.pyplot as plt

#pts1 = np.array([(669, 639), (472, 586), (709, 428), (761, 395), (879, 313), (953, 264), (334, 687), (644, 473), (804, 365), (909, 297), (615, 562), (671, 519), (721, 479), (757, 451), (794, 423)])
#pts2 = np.array([(104, 46), (165, 53), (164, 134), (162, 162), (156, 271), (156, 389), (163, 23), (163, 102), (158, 196), (156, 305), (140, 69), (139, 87), (137, 105), (137, 128), (137, 146)])
pts1 = np.array([(141, 285), (169, 346), (212, 388), (280, 414), (303, 240), (318, 52), (351, 61), (382, 73), (441, 109), (145, 163), (176, 122)])
pts2 = np.array([(376, 251), (342, 409), (340, 452), (370, 582), (643, 413), (893, 220), (928, 291), (941, 326), (950, 495), (516, 135), (730, 122)])

class HeatMap:
  def __init__(self, map_img_path):
    self.heat_points = {}
    self.points_list = []
    self.map_img = cv2.imread(map_img_path, 0)
    self.wrap_map_img = None
    self.calc_homography_matrix(pts1, pts2)

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
      self.points_list.append(center)
      ret = self.filter_points(center[0], center[1])
      if ret is not None:
        if center in self.heat_points.keys():
          self.heat_points[center] += 1
        else:
          self.heat_points[center] = 1

  def draw_heatmap(self, q):
    while True:
      points_list = q.get()
      if points_list:
        xi, yi, zi, x, y = self.calc_gaussian_kde(points_list)
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, shading='auto')
        #plt.xlim(x.min(), x.max())
        #plt.ylim(y.max(), y.min())
        #plt.imshow(self.map_img)
        #plt.show()
        #plt.pause(0.05)
        
  def calc_gaussian_kde(self, points_list):
    x, y = np.array(points_list).T
    # gaussian kernel density estimation
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    return xi, yi, zi, x, y
    
  def calc_homography_matrix(self, pts1, pts2):
    # find homography
    self.M, _ = cv2.findHomography(pts1, pts2)

  def apply_matrix(self, point):
    # apply matrix to point
    pt = np.array((point[0], point[1], 1)).reshape(3, 1) 
    px, py, pz = self.M.dot(pt)
    # convert homogeneous to cartesian
    px = np.int(np.round(px/pz))
    py = np.int(np.round(py/pz))
    return px, py
      
  def draw_heatpoints(self, frame):
    """
    Draw heat points on frame
    """
    for p in self.heat_points:
      frame = cv2.circle(frame, p, radius=2, color=(0, 0, 255), thickness=-1)
    return frame
  
  def filter_points(self, x, y):
    if self.wrap_map_img is None:
      self.wrap_map_img = cv2.warpPerspective(self.map_img, self.M, (1280, 720))
    px, py = self.apply_matrix((x, y))
    if (px < 0 or px > self.wrap_map_img.shape[1]): return None
    if (py < 0 or py > self.wrap_map_img.shape[0]): return None
    if self.wrap_map_img[py, px] == 255:
      return x, y
    else:
      return None
        
