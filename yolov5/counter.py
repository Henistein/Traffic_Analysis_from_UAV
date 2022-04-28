import numpy as np

"""
class Counter:
  def __init__(self):
    self.lines_in = []
    self.lines_out = []
    self.centers = []

  def add_line_in(self, p1, p2):
    self.lines_in.append([p1, p2])

  def add_line_out(self, p1, p2):
    self.lines_out.append([p1, p2])


  def check_intersection(self, centers):
    for x,y in centers:
      # check in
      for li in lines_in:
        if 
    
"""
class Box:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.counter = 0
        self.frame_countdown = 0
    def overlap(self, start_point, end_point):
        if self.start_point[0] >= end_point[0] or self.end_point[0] <= start_point[0] or \
                self.start_point[1] >= end_point[1] or self.end_point[1] <= start_point[1]:
            return False
        else:
            return True
