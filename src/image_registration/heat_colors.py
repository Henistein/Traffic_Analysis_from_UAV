import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
np.random.seed(20)

img = cv2.imread('rua.png')
heatmap = np.zeros(img.shape)

points = np.array([(102, 46), (165, 53), (163, 132), (163, 101), (162, 161), (158, 195), (159, 271), (156, 306), (155, 390), (153, 437), (154, 541), (295, 161), (223, 140), (139, 68), (139, 87), (137, 105), (136, 123), (136, 146), (134, 169), (134, 193), (133, 214), (132, 240), (131, 263), (129, 289), (184, 102), (184, 120), (184, 140)])


# DEBUG
"""
for point in points:
  w,h = point
  img = cv2.circle(img, (w,h), radius=5, color=(0, 255, 255), thickness=-1)

cv2.imshow('frame', img)
cv2.waitKey(0)
"""

# multiply some point occurrencies
points_dict = {}
points_list = []
for point in points:
  occ = np.random.randint(0, 400)
  points_dict[tuple(point)] = occ
  for i in range(occ):
    points_list.append(tuple(point))


# heatmap
from scipy.stats.kde import gaussian_kde
from PIL import Image
import io
img = plt.imread('rua.png')

points_list = np.array(points_list)
x, y = points_list[:, 0], points_list[:, 1]
#x,y = np.array(list(points_dict.keys())).T
#weights = np.array(list(points_dict.values()))

k = gaussian_kde(np.vstack([x, y]))
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# alpha=0.5 will make the plots semitransparent
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, shading='auto')
plt.xlim(x.min(), x.max())
plt.ylim(y.max(), y.min())
plt.imshow(img)
plt.show()

