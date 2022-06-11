import numpy as np
import geopy.distance

class SpeedHandler:
  def __init__(self):
    self.speeds = {}
    self.smoothed_speeds = {}
    k = 50
    self._kern = np.ones(2*k+1)/(2*k+1)

  def update_speeds(self, pts, last_pts):
    for id_ in set(pts).intersection(set(last_pts)):
      # calculate euclidean distance
      dist = geopy.distance.geodesic(last_pts[id_], pts[id_]).meters
      # calculate speed by dividing distance by time
      if id_ in self.speeds.keys():
        self.speeds[id_].append((dist/0.1)*3.6)
      else:
        self.speeds[id_] = [(dist/0.1)*3.6]

  def smooth_speeds(self):
    self.smoothed_speeds = {}
    for id_ in self.speeds.keys():
      self.smoothed_speeds[id_] = np.convolve(self.speeds[id_], self._kern, mode='same') 
