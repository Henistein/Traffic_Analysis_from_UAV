import cv2
from multiprocessing import Process

class DrawPoint:
  def __init__(self, img):
    self.img = img
    self.last_coords = None
    self.points = []

  def extract_coordinates(self, event, x, y, flags, parameters):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.last_coords = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
      if self.last_coords is not None:
        # Draw point
        self.img = cv2.circle(self.img, self.last_coords, radius=2, color=(0, 0, 255), thickness=-1)
        self.points.append(self.last_coords)
  
  @property
  def image(self):
    return self.img

def annotate_image(img):
  img = cv2.imread(img)
  dp = DrawPoint(img)
  cv2.namedWindow('image')
  cv2.setMouseCallback('image', dp.extract_coordinates)

  while True:
    cv2.imshow('image', dp.image)
    if cv2.waitKey(1) == ord('q'):
      cv2.destroyAllWindows()
      break
  #return dp.points
  print(dp.points)

if __name__ == '__main__':
  # annotate maps image
  maps_image = 'rua.png'
  #maps_img_pts = annotate_image(maps_image)
  #print(maps_img_pts)

  # annotate dron image
  drone_image = 'drone.png'
  #drone_img_pts = annotate_image(drone_image)
  #print(drone_img_pts)
  p1 = Process(target=annotate_image, args=(maps_image,))
  p2 = Process(target=annotate_image, args=(drone_image,))

  p1.start()
  p2.start()
  p1.join()
  p2.join()
