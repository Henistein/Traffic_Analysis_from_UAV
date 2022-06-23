# count how many cars crossed a given line

import cv2
import numpy as np

class Counter:
  def __init__(self, classes):
    self.img = None
    self.line = {"big":None, "small":np.array([[648, 180], [720, 233]], np.int32)}
    self.box = {"big":None, "small":np.array([[675, 174], [721, 215]], np.int32)}
    self.idpoints = {}
    self.counter = 0
    self.avg_speed = []
    self.stats = {classes.index(k):0 for k in classes}
    self.classes = classes
  
  def draw_line(self):
    cv2.polylines(self.img, [self.line["small"]], False, (0,255,0))
  
  def draw_box(self):
    cv2.rectangle(self.img, tuple(self.box["small"][0]), tuple(self.box["small"][1]), (0,255,0), 2)
  
  def get_ids_inside_box(self, curr_centers):
    ids = []
    for id in curr_centers:
      center = curr_centers[id]
      # check if center is inside box
      if self.box["big"][0][0] <= center[0] <= self.box["big"][1][0] and self.box["big"][0][1] <= center[1] <= self.box["big"][1][1]:
        ids.append(id)
    return ids
    
  
  def update_img(self, img, scale):
    self.img = img
    self.scale = scale
    self.line["big"] = np.array([[int(self.line["small"][0][0]*scale[0]), int(self.line["small"][0][1]*scale[1])], [int(self.line["small"][1][0]*scale[0]), int(self.line["small"][1][1]*scale[1])]], np.int32)
    self.box["big"] = np.array([[int(self.box["small"][0][0]*scale[0]), int(self.box["small"][0][1]*scale[1])], [int(self.box["small"][1][0]*scale[0]), int(self.box["small"][1][1]*scale[1])]], np.int32)
  
  def check_intersection(self, line1, line2):
    def ccw(A, B, C):
      return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    A,B = line1[0], line1[1]
    C,D = line2[0], line2[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

  
  def count(self, curr_centers, id_clss):
    self.draw_line()
    self.draw_box()
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
        #print(id)
        #print(id_clss)
        self.stats[float(id_clss[float(id)])] += 1
        del self.idpoints[id]

  def show_stats(self):
    print(f"Count: {self.counter}")
    for n in self.classes:
      print(f"{n}: {self.stats[self.classes.index(n)]}")

    
  def get_average_speed_in_box(self, curr_centers, speeds):
    ids_inside = self.get_ids_inside_box(curr_centers)
    speeds_inside = []
    for id in ids_inside:
      if id in speeds:
        speeds_inside.append(speeds[id][-1])
    if len(speeds_inside) > 0:
      self.avg_speed.append(np.mean(speeds_inside))
    if len(self.avg_speed) > 0:
      return np.mean(self.avg_speed)
  
  def manually_draw_line(self):
    # let user draw line in self.img
    cv2.namedWindow('draw_line')
