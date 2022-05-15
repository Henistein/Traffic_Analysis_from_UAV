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
    # ration matrix
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))

    rotated_pts = []
    for pt in pts:
      # subtract the center of the sqare
      pt = (pt[0]-center[0], pt[1]-center[1])
      # apply rotation matrix
      pt = R@pt
      # add the center of the square
      pt = (int(pt[0]+center[0]), int(pt[1]+center[1]))

      rotated_pts.append(pt)

    return rotated_pts

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

    return img_crop, img_rot, angle

  def draw_compass_head(self, center, angle, img, cam_w, cam_h):
    scale = [3.75, 3.865]
    #sizeX,sizeY = 1280/scale[0]/2, 720/scale[1]/2
    sizeX,sizeY = int(cam_w), int(cam_h)

    p1 = [center[0]-sizeX, center[1]-sizeY]
    p2 = [center[0]+sizeX, center[1]-sizeY]
    p3 = [center[0]-sizeX, center[1]+sizeY]
    p4 = [center[0]+sizeX, center[1]+sizeY]

    p1,p2,p3,p4 = self.rotate_camera(center,angle, [p1,p2,p3,p4])

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
    
    img_crop, _, crop_angle = self._crop_rect(self.image,rect)

    # rotate image to match drone perspective

    if angle >= 270 or angle == 0:
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif angle >= 180:
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle >= 90:
      img_crop = cv2.rotate(img_crop, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img,img_crop,crop_angle


  
if __name__ == '__main__':
  """
  mapp = MAP(
      lat_north=40.275098,
      long_west=-7.509555,
      lat_south=40.267330,
      long_east=-7.493187,
      lat_center=40.271243,
      long_center=-7.501679,
      image=cv2.imread('MAPA2.jpg')
  )
  """
  mapp = MAP(
      lat_north=40.273668,
      long_west=-7.506679,
      lat_south=40.268085,
      long_east=-7.493513,
      lat_center=40.271018,
      long_center=-7.500335,
      image=cv2.imread('MAPA.jpg')
  )

  scale = [3.75, 3.865]
  w,h = int(mapp.image.shape[1]/scale[0]), int(mapp.image.shape[0]/scale[1])
  small_img = cv2.resize(mapp.image, (w,h), cv2.INTER_AREA)

  data = pd.read_csv('filtered_data.csv')
  latitude_list = data["latitude"].tolist()
  longitude_list = data["longitude"].tolist()
  compass_heading_list = data[" compass_heading(degrees)"].tolist()

  video_points = [(533, 306), (577, 51), (478, 400), (500, 437),
                  (530, 367), (821, 452), (895, 419), (1033, 295)]

  for i in range(len(latitude_list)):
    img = small_img.copy()
    lat = latitude_list[i]
    lon = longitude_list[i]
    angle = compass_heading_list[i]
    # calculate new_lat and new_lon
    dw = 134.4/2
    dh = 75.6/2
    new_lat = lat + (dh/(6378*1000))*(180/np.pi)
    new_lon = lon + (dw/(6378*1000))*(180/np.pi)/np.cos(lat*np.pi/180)
    # convert new lat and new lon
    nx,ny = mapp.coords_to_pixels(new_lat,new_lon)
    nx,ny = nx/scale[0],ny/scale[1]
    nx,ny = int(nx),int(ny)
    # convert to xy
    x,y = mapp.coords_to_pixels(lat,lon)
    # scale xy
    x,y = x/scale[0],y/scale[1]
    x,y = int(x),int(y)
    # paint xy on small image
    img = cv2.circle(img,(x,y),radius=3,color=(0,255,0),thickness=-1)
    # calculate width and height of camera
    cam_w, cam_h = abs(x-nx), abs(y-ny)

    # draw compass head (rectangle)
    img,img_crop, crop_angle = mapp.draw_compass_head((x,y),angle,img, cam_w, cam_h)

    # convert video points to small image
    # scale points from 1280x720 to small image dimensions
    scale_x, scale_y = 1280/(cam_w*2), 720/(cam_h*2)
    scaled_pts = []
    for pt in video_points:
      scaled_pts.append((int(pt[0]/scale_x), int(pt[1]/scale_y)))
    
    # convert point coordinates to map image
    big_image_dim = mapp.image.shape[1]//scale[0], mapp.image.shape[0]//scale[1]
    #incx, incy = (big_image_dim[0]//2 - cam_w), (big_image_dim[1]//2 - cam_h)
    incx, incy = x-cam_w, y-cam_h

    for i in range(len(scaled_pts)):
      pt = scaled_pts[i]
      pt = (int(pt[0]+incx), int(pt[1]+incy))
      pt = mapp.rotate_camera((x,y), angle, [pt])[0]
      scaled_pts[i] = pt

      # draw point on image    
      img = cv2.circle(img,scaled_pts[i],radius=3,color=(0,255,0),thickness=-1)



    cv2.imshow('frame1', img)
    cv2.imshow('frame2', img_crop)
    cv2.waitKey(0)
