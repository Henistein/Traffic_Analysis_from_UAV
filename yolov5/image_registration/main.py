import numpy as np
import cv2
from matplotlib import pyplot as plt

# open the image files
img1 = cv2.imread('drone.png', 0)
img2 = cv2.imread('rua.png', 0)

pts1 = np.array([(637, 445), (314, 495), (448, 396), (613, 289), (675, 242), (723, 214), (768, 184), (835, 133), (865, 115), (796, 258)])
pts2 = np.array([(103, 45), (163, 23), (165, 52), (162, 101), (163, 134), (162, 162), (160, 196), (155, 269), (157, 306), (116, 144)])

# find homography
M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# load points
points = np.loadtxt('points.txt', delimiter=' ')

for p in points:
  # homogeneous point
  pt = np.array((p[0],p[1],1)).reshape((3,1))
  px, py, pz = M.dot(pt)
  # convert homogeneous to cartesian
  px = np.int(np.round(px/pz))
  py = np.int(np.round(py/pz))

  img2 = cv2.circle(img2, (px, py), radius=2, color=(0, 255, 255), thickness=-1)

cv2.imshow('image', img2)
cv2.waitKey(0)

# use homography
"""
h, w = img2.shape
imgReg = cv2.warpPerspective(img1, M, (w, h))
cv2.imshow('image', imgReg)
cv2.waitKey(0)
"""

