import cv2
import numpy as np

class HeatMap:
  def __init__(self):
    self.heat_points = []

  def update_points(self, bboxes):
    """
    Extract centers from the bboxes and update the points
    """
    for det in bboxes:
      x1,y1,x2,y2 = det
      center = (int((x1+x2)/2), int((y1+y2)/2))
      self.heat_points.append(center)
      
  def draw_center(self, frame):
    """
    Draw heat points on frame
    """
    for p in self.heat_points:
      frame = cv2.circle(frame, p, radius=2, color=(0, 0, 255), thickness=-1)
    return frame
