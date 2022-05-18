import numpy as np
import cv2
import pandas as pd
from osgeo import osr, ogr, gdal

class GeoRef:
  # https://stackoverflow.com/questions/58623254/find-pixel-coordinates-from-lat-long-point-in-geotiff-using-python-and-gdal
  def __init__(self, tif_image):
    self.image = cv2.imread(tif_image)
    self.ds = gdal.Open(tif_image)
    self.target = osr.SpatialReference(wkt=self.ds.GetProjection())
    self.source = osr.SpatialReference()
    self.source.ImportFromEPSG(4326)
    self.transform = osr.CoordinateTransformation(self.source, self.target)
    self.geo_matrix = self.ds.GetGeoTransform()

  def _coords_to_pixels(self, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ul_x= self.geo_matrix[0]
    ul_y = self.geo_matrix[3]
    x_dist = self.geo_matrix[1]
    y_dist = self.geo_matrix[5]
    pixel = int((x - ul_x) / x_dist)
    line = -int((ul_y - y) / y_dist)
    return pixel, line

  def coords_to_pixels(self, lat, lon):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(lon, lat)
    point.Transform(self.transform)
    return self._coords_to_pixels(point.GetX(), point.GetY())

class GeoInterpolation:
  def __init__(self, lat_north, long_west, lat_south, long_east, lat_center, long_center, image):
    # image
    self.image = image
    self.width = self.image.shape[1]
    self.height = self.image.shape[0]
    # coordinates
    self.lat_north = lat_north
    self.long_west = long_west
    self.lat_south = lat_south
    self.long_east = long_east
    self.lat_center = lat_center
    self.long_center = long_center
    # adjust
    self.adjx, self.adjy = self.find_adjust_scale()

  def coords_to_pixels(self, lat, lon):
    """
    https://stackoverflow.com/questions/2103924/mercator-longitude-and-latitude-calculations-to-x-and-y-on-a-cropped-map-of-the/10401734#10401734
    """
    mapLatBottomRad = self.lat_south * np.pi / 180
    latitudeRad = lat * np.pi / 180
    mapLngDelta = (self.long_east - self.long_west)

    worldMapWidth = ((self.width / mapLngDelta) * 360) / (2 * np.pi)
    mapOffsetY = (worldMapWidth / 2 * np.log((1 + np.sin(mapLatBottomRad)) / (1 - np.sin(mapLatBottomRad))))

    x = (lon - self.long_west) * (self.width / mapLngDelta)
    y = self.height - ((worldMapWidth / 2 * np.log((1 + np.sin(latitudeRad)) / (1 - np.sin(latitudeRad)))) - mapOffsetY)

    if not hasattr(self, "adjx"):
      return x,y
    return self.adjx*x, self.adjy*y

  def find_adjust_scale(self):
    cx, cy = self.width//2, self.height//2
    x,y = self.coords_to_pixels(self.lat_center, self.long_center)
    return cx/x, cy/y # adjx, adjy

class MapDrone:
  def __init__(self, geo):
    scale = [3.75, 3.865]
    self.geo = geo
    self.image = self.geo.image
    #w,h = int(geo.image.shape[1]/scale[0]), int(geo.image.shape[0]/scale[1])
    #small_img = cv2.resize(self.image, (w,h), cv2.INTER_AREA)

    data = pd.read_csv('filtered_data.csv')

    self.latitude_list = data["latitude"][:2800].tolist()
    self.longitude_list = data["longitude"][:2800].tolist()
    self.compass_heading_list = data[" compass_heading(degrees)"][:2800].tolist()
    #self.small_img = small_img

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

  def _degrees_to_rads(self, angle):
    return np.pi*angle/180

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
  def get_next_data(self, k, idcenters):
    img = self.geo.image.copy()
    lat = self.latitude_list[k]
    lon = self.longitude_list[k]
    angle = self.compass_heading_list[k]
    # convert to xy
    x,y = self.geo.coords_to_pixels(lat,lon)
    # paint xy on map image
    img = cv2.circle(img,(x,y),radius=3,color=(0,0,255),thickness=2)

    # calculate distance from center to idcenters
    gsdc = 10.546875
    center = (1280/2, 720/2)
    scaled_pts = {}
    for k,pt in idcenters.items():
      # Rotate point
      pt = self.rotate_camera(center, angle, [pt])[0]
      dist_cx_x = abs(pt[0]-center[0]) * gsdc / 100
      dist_cy_y = abs(pt[1]-center[1]) * gsdc / 100
      new_lat = lat + (dist_cy_y/(6378*1000))*(180/np.pi)
      new_lon = lon + (dist_cx_x/(6378*1000))*(180/np.pi)/np.cos(lat*np.pi/180)
      # convert lat and lon to pixels
      x,y = self.geo.coords_to_pixels(new_lat,new_lon)
      scaled_pts[k] = (x,y)
      # draw points on image
      img = cv2.circle(img,(int(x),int(y)),radius=3,color=(0,0,255),thickness=2)

    return img,scaled_pts
