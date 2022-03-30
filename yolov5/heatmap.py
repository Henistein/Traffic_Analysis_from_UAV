import cv2
import numpy as np

pts1 = np.array([(669, 639), (472, 586), (709, 428), (761, 395), (879, 313), (953, 264), (334, 687), (644, 473), (804, 365), (909, 297), (615, 562), (671, 519), (721, 479), (757, 451), (794, 423)])
pts2 = np.array([(104, 46), (165, 53), (164, 134), (162, 162), (156, 271), (156, 389), (163, 23), (163, 102), (158, 196), (156, 305), (140, 69), (139, 87), (137, 105), (137, 128), (137, 146)])

class HeatMap:
  def __init__(self, map_img_path):
    self.heat_points = {}
    self.points_list = []
    self.map_img = cv2.imread(map_img_path)

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
      if center in self.heat_points.keys():
        self.heat_points[center] += 1
      else:
        self.heat_points[center] = 1

  def draw_heatmap(self):
    while True:
      if len(self.points_list):
        print('TOU')
        xi, yi, zi, x, y = self.draw_heatmap()
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, shading='auto')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.max(), y.min())
        plt.imshow(self.map_img)
        plt.pause(0.05)

  def calc_gaussian_kde(self):
    x, y = np.array(self.points_list).T
    # gaussian kernel density estimation
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    return xi, yi, zi, x, y
    
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
