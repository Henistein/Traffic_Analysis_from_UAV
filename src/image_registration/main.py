import numpy as np
import cv2
from matplotlib import pyplot as plt

# open the image files
img1 = cv2.imread('real_drone.png', 0)
img2 = cv2.imread('rua.png', 0)

pts1 = np.array([(668, 639), (472, 585), (710, 424), (644, 476), (759, 396), (807, 364), (876, 313), (907, 299), (953, 264), (975, 249), (1005, 227), (503, 353), (592, 389), (617, 561), (672, 519), (719, 481), (757, 453), (795, 423), (825, 398), (855, 376), (877, 357), (899, 340), (920, 323), (939, 309), (601, 455), (645, 427), (686, 404)])
pts2 = np.array([(102, 46), (165, 53), (163, 132), (163, 101), (162, 161), (158, 195), (159, 271), (156, 306), (155, 390), (153, 437), (154, 541), (295, 161), (223, 140), (139, 68), (139, 87), (137, 105), (136, 123), (136, 146), (134, 169), (134, 193), (133, 214), (132, 240), (131, 263), (129, 289), (184, 102), (184, 120), (184, 140)])

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
h, w = img2.shape
imgReg = cv2.warpPerspective(img1, M, (w, h))
cv2.imshow('image', imgReg)
cv2.waitKey(0)

