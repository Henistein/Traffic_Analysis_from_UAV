import cv2
import numpy as np

seq1 = [(591, 455), (745, 463), (745, 421), (813, 438), (609, 589), (581, 589), (603, 631), (588, 631), (596, 285), (379, 427), (379, 467), (650, 129), (541, 84), (1003, 280), (959, 272), (973, 317)]
seq1 = np.array(seq1)
seq2 = [(702, 686), (597, 640), (566, 597), (651, 569), (652, 318), (675, 105), (501, 332), (549, 223), (568, 209)]
seq2 = np.array(seq2)

map1 = [(846, 870), (944, 873), (943, 848), (996, 855), (854, 966), (830, 966), (848, 1000), (837, 1000), (840, 755), (697, 853), (700, 885), (877, 648), (802, 615), (1120, 748), (1091, 741), (1100, 775)]
map1 = np.array(map1)
map2 = [(876, 649), (804, 617), (784, 594), (837, 569), (835, 405), (847, 265), (739, 411), (767, 341), (777, 333)]
map2 = np.array(map2)

H1, _ = cv2.findHomography(seq1, map1, cv2.RANSAC, 5.0)
H2, _ = cv2.findHomography(seq2, map2, cv2.RANSAC, 5.0)

img1 = cv2.imread('images/seq001.png')
img2 = cv2.imread('images/seq002.png')

img_w1 = cv2.warpPerspective(img1, H1, (1920,1080))
img_w2 = cv2.warpPerspective(img2, H2, (1920,1080))

cv2.imshow('frame',img_w1)
cv2.imshow('frame',img_w2)
cv2.waitKey(0)

"""
if __name__ == '__main__':
  cap = cv2.VideoCapture('drone.mp4')
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret != True:
      break
"""
