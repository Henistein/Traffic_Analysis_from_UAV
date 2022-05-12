import cv2
import numpy as np
import pandas as pd
import time

class MAP:
  def __init__(self, lat_north, long_west, lat_south, long_east, lat_center, long_center, image):
    # image
    self.image = image
    self.width = image.shape[1]
    self.height = image.shape[0]
    # coordinates
    self.lat_north = lat_north
    self.long_west = long_west 
    self.lat_south = lat_south 
    self.long_east = long_east 
    self.lat_center = lat_center
    self.long_center = long_center
    # calculate pixel per lat
    self.pixels_per_lat = self.height/(self.lat_north-self.lat_south)
    self.pixels_per_long = self.width/(self.long_east-self.long_west)
    # adjust
    adjx, adjy = self.find_adjust_scale()
    self.pixels_per_lat *= adjx
    self.pixels_per_long *= adjy

  def coords_to_pixels(self, lat, lon):
    x = (lon-self.long_west)*self.pixels_per_long
    y = abs(lat-self.lat_north)*self.pixels_per_lat
    return x,y

  def find_adjust_scale(self):
    cx, cy = self.width//2, self.height//2
    x,y = self.coords_to_pixels(self.lat_center, self.long_center)
    return cx/x, cy/y # adjx, adjy

  def _degrees_to_rads(self, angle):
    return np.pi*angle/180

  def rotate_camera(self, center, angle, pts):
    angle = self._degrees_to_rads(angle)
    p1,p2,p3,p4 = pts
    # subtract of the center of triangle
    p1 = (p1[0]-center[0], p1[1]-center[1])
    p2 = (p2[0]-center[0], p2[1]-center[1])
    p3 = (p3[0]-center[0], p3[1]-center[1])
    p4 = (p4[0]-center[0], p4[1]-center[1])

    # ration matrix
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    p1 = R@p1
    p2 = R@p2
    p3 = R@p3
    p4 = R@p4
    
    # add of the center of triangle
    p1 = (int(p1[0]+center[0]), int(p1[1]+center[1]))
    p2 = (int(p2[0]+center[0]), int(p2[1]+center[1]))
    p3 = (int(p3[0]+center[0]), int(p3[1]+center[1]))
    p4 = (int(p4[0]+center[0]), int(p4[1]+center[1]))

    return p1,p2,p3,p4

  def _crop_rect(self,img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

  def draw_compass_head(self, center, angle, img):
    scale = [3.75, 3.865]
    sizeX,sizeY = 1280/scale[0]/4, 720/scale[1]/4

    p1 = [center[0]-sizeX, center[1]-sizeY]
    p2 = [center[0]+sizeX, center[1]-sizeY]
    p3 = [center[0]-sizeX, center[1]+sizeY]
    p4 = [center[0]+sizeX, center[1]+sizeY]

    p1,p2,p3,p4 = self.rotate_camera(center,angle, (p1,p2,p3,p4))

    cnt = np.array([p1,p2,p3,p4])
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # draw camera on small image
    cv2.drawContours(img,[box],0,(0,0,255),2)

    # get image crop from big image
    rect = list(rect)
    rect[0] = list(rect[0])
    rect[1] = list(rect[1])
    rect[0][0] = rect[0][0]*scale[0]
    rect[1][0] = rect[1][0]*scale[0]
    rect[0][1] = rect[0][1]*scale[1]
    rect[1][1] = rect[1][1]*scale[1]
    
    img_crop, _ = self._crop_rect(self.image,rect)

    # rotate image to match drone perspective

    if angle >= 270 or angle == 0:
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif angle >= 180:
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle >= 90:
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img,img_crop

class Teste:
  def __init__(self, mapp):
    scale = [3.75, 3.865]
    w,h = int(mapp.image.shape[1]/scale[0]), int(mapp.image.shape[0]/scale[1])
    small_img = cv2.resize(mapp.image, (w,h), cv2.INTER_AREA)

    data = pd.read_csv('filtered_data.csv')

    self.latitude_list = data["latitude"][:2800].tolist()
    self.longitude_list = data["longitude"][:2800].tolist()
    self.compass_heading_list = data[" compass_heading(degrees)"][:2800].tolist()
    self.small_img = small_img
    self.mapp = mapp

  def get_next_data(self, k):
    scale = [3.75, 3.865]
    img = self.small_img.copy()
    lat = self.latitude_list[k]
    lon = self.longitude_list[k]
    angle = self.compass_heading_list[k]
    # convert to xy
    x,y = self.mapp.coords_to_pixels(lat,lon)
    # scale xy
    x,y = x/scale[0],y/scale[1]
    x,y = int(x),int(y)
    # paint xy on small image
    img = cv2.circle(img,(x,y),radius=3,color=(0,255,0),thickness=-1)
    # draw compass head (triangle)
    img,img_crop = self.mapp.draw_compass_head((x,y),angle,img)

    return img,img_crop


  
if __name__ == '__main__':
  cap = cv2.VideoCapture('moving_drone.mp4')
  print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


  mapp = MAP(
      lat_north=40.275098,
      long_west=-7.509555,
      lat_south=40.267330,
      long_east=-7.493187,
      lat_center=40.271243,
      long_center=-7.501679,
      image=cv2.imread('MAPA2.jpg')
  )


  scale = [3.75, 3.865]
  w,h = int(mapp.image.shape[1]/scale[0]), int(mapp.image.shape[0]/scale[1])
  small_img = cv2.resize(mapp.image, (w,h), cv2.INTER_AREA)

  data = pd.read_csv('filtered_data.csv')

  latitude_list = data["latitude"][:2800].tolist()
  longitude_list = data["longitude"][:2800].tolist()
  compass_heading_list = data[" compass_heading(degrees)"][:2800].tolist()

  print(len(latitude_list))

  # video 8400 data 2800
  k = 0
  for i in range(8400):
    ret, frame = cap.read()
    if ret is False: exit(0)

    if i%3 == 0:
      img,img_crop = get_next_data(k)  
      k += 1
      cv2.imshow('map image', img)
      cv2.imshow('crop image', img_crop)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
