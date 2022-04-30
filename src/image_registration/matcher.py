import numpy as np
import cv2
from matplotlib import pyplot as plt

# open the image files
img1 = cv2.imread('drone.png', 0)
img2 = cv2.imread('rua.png', 0)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Match descriptors.
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
  if m.distance < 0.75*n.distance:
    good.append([m])

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.show()
