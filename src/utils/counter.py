# count how many cars crossed a given line

import cv2
import numpy as np

class Counter:
  def __init__(self):
    self.img = None
    self.line = {"big":None, "small":np.array([[620, 200], [700, 287]], np.int32)}
    self.counter = 0
    self.idpoints = {}
  
  def draw_line(self):
    cv2.polylines(self.img, [self.line["small"]], False, (0,255,0))
  
  def update_img(self, img, scale):
    self.img = img
    self.scale = scale
    self.line["big"] = np.array([[int(self.line["small"][0][0]*scale[0]), int(self.line["small"][0][1]*scale[1])], [int(self.line["small"][1][0]*scale[0]), int(self.line["small"][1][1]*scale[1])]], np.int32)
  
  def check_intersection(self, line1, line2):
    def ccw(A, B, C):
      return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    A,B = line1[0], line1[1]
    C,D = line2[0], line2[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

  
  def count(self, curr_centers):
    self.draw_line()
    for id in curr_centers:
      if id not in self.idpoints:
        self.idpoints[id] = []
      self.idpoints[id].append(curr_centers[id])
      # get last point and first point
      p1 = self.idpoints[id][-1]
      p2 = self.idpoints[id][0]
      cv2.line(self.img, (int(p1[0]/self.scale[0]),int(p1[1]/self.scale[1])), (int(p2[0]/self.scale[0]),int(p2[1]/self.scale[1])), (0,0,255), 2)
      # create line with p1 and p2
      line = np.array([p1, p2], np.int32)
      # check if line and self.line intersect
      if self.check_intersection(line, self.line["big"]):
        self.counter += 1
        del self.idpoints[id]
    print(self.counter)
  
  def manually_draw_line(self):
    # let user draw line in self.img
    cv2.namedWindow('draw_line')