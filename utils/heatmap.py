import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

class HeatMap:
  def __init__(self, image):
    self.image = image
    self.points = []

  def update_points(self, pt):
    self.points.append(pt)

  def draw_heatmap(self):
    x, y = np.array(self.points).T
    # gaussian kernel density estimation
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    plt.figure(figsize=(1280/72, 720/72))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, shading='auto')
    plt.imshow(self.image)
    plt.savefig('heatmap.png')