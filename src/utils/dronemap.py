import math
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
  FRAME_WIDTH = 1280
  FRAME_HEIGHT = 720
  FOOTPRINT_WIDTH = 200
  FOOTPRINT_HEIGHT = 120
  def __init__(self, geo):
    self.geo = geo
    self.map = self.geo.image
    # resized map (1280x720)
    self.scale = [self.map.shape[1]/self.FRAME_WIDTH, self.map.shape[0]/self.FRAME_HEIGHT]
    w,h = int(geo.image.shape[1]/self.scale[0]), int(geo.image.shape[0]/self.scale[1])
    self.resized_map = cv2.resize(self.map, (w,h), cv2.INTER_AREA)
    # extract drone data
    data = pd.read_csv('filtered_data.csv')
    self.latitude_list = data["latitude"][:2800].tolist()
    self.longitude_list = data["longitude"][:2800].tolist()
    self.compass_heading_list = data[" compass_heading(degrees)"][:2800].tolist()

  def rotate_camera(self, center, angle, pts):
    """
    :param center: center of the rotation point (drone current position)
    :param angle: angle of rotation (drone heading)
    :param pts: list of points to rotate
    @return: list of rotated points
    """
    angle = math.radians(angle)
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
    """
    Crop the image to the defined rectangle
    :param img: image to crop 
    :param rect: crop rectangle
    @return: cropped image
    """
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
  
  def _add_distance_to_coordinates(self, lat, lon, dist_y, dist_x):
    new_lat = lat + (dist_y/(6378*1000))*(180/np.pi)
    new_lon = lon + (dist_x/(6378*1000))*(180/np.pi)/np.cos(lat*np.pi/180)
    return new_lat, new_lon

  def draw_drone_footprint(self, center, angle, img, cam_w, cam_h):
    """
    Draws drone footprint on the map
    Extracts drone footprint from the map
    :param center: drone current position
    :param angle: drone heading
    :param img: map image
    :param cam_w: width of the frame from the center to the edge
    :param cam_h: height of the frame from the center to the edge
    @return: map image with drone footprint
    @return: footprint image
    """
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
    rect[0][0] = rect[0][0]*self.scale[0]
    rect[1][0] = rect[1][0]*self.scale[0]
    rect[0][1] = rect[0][1]*self.scale[1]
    rect[1][1] = rect[1][1]*self.scale[1]
    
    footprint,_,_ = self._crop_rect(self.map,rect)

    # rotate image to match drone perspective
    if angle >= 270 or angle == 0:
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif angle >= 180:
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle >= 90:
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img,footprint

  def get_data(self, k):
    """
    Get data from the list
    :param k: index of the data
    @return: latitude, longitude, compass_heading
    """
    return self.latitude_list[k], self.longitude_list[k], self.compass_heading_list[k]

  def get_next_data(self, k, idcenters):
    """
    Process the data from the drone
    Extract the drone footprint from the map
    Draw the points from the drone frame to the map
    :param k: index of the drone data list
    :param idcenters: list frame object centers
    """
    resized_map = self.resized_map.copy()
    lat,lon,heading = self.get_data(k)

    # convert drone current position to map coordinates
    x,y = self.geo.coords_to_pixels(lat,lon)
    # scale xy to resized map
    rx,ry = int(x/self.scale[0]),int(y/self.scale[1])
    # draw xy on the resized map
    resized_map = cv2.circle(resized_map,(rx,ry),radius=3,color=(0,0,255),thickness=-1)

    """
    - Calculate latitudes and longitudes of the drone footprint
    - Convert them to pixels
    - Scale them to the resized map
    VIDEO_FRAME -> FOOTPRINT -> MAP -> RESIZED_MAP
    """
    ftp_lat, ftp_lon = self._add_distance_to_coordinates(
      lat, lon, 
      MapDrone.FOOTPRINT_HEIGHT/2, MapDrone.FOOTPRINT_WIDTH/2
    )
    # convert ftp_lat and ftp_lon to map coordinates
    ftp_x,ftp_y = self.geo.coords_to_pixels(ftp_lat,ftp_lon)
    # scale ftp_x and ftp_y to resized map
    ftp_x,ftp_y = int(ftp_x/self.scale[0]),int(ftp_y/self.scale[1])

    # calculate pixel distance from the center of the frame to the edge (resized map)
    dist_w, dist_h = abs(rx-ftp_x), abs(ry-ftp_y)
    # draw footprint on the resized map
    resized_map,footprint = self.draw_drone_footprint((rx,ry),heading,resized_map,dist_w,dist_h)

    """
    Convert video frame points to map coordinates
    - Scale video frame points to cropped footprint
    """
    # scale used to map video frame points to cropped footprint
    ftp_scale_x = MapDrone.FRAME_WIDTH/(dist_w*2)
    ftp_scale_y = MapDrone.FRAME_HEIGHT/(dist_h*2)
    # auxiliar variables to convert from footprint to map coordinates
    incx, incy = rx-dist_w, ry-dist_h

    # convert video frame points to cropped footprint and then to map coordinates
    scaled_pts = {}
    for k,pt in idcenters.items():
      # scaling the video frame points to cropped footprint 
      pt = (pt[0]/ftp_scale_x, pt[1]/ftp_scale_y)
      # convert from cropped footprint to map coordinates
      pt = (pt[0]+incx, pt[1]+incy)
      # rotate points to match drone heading
      pt = self.rotate_camera((rx,ry), heading, [pt])[0]
      # draw point on resized map
      resized_map = cv2.circle(resized_map,(int(pt[0]),int(pt[1])),radius=3,color=(0,255,0),thickness=-1)
      # save points in original map scale
      scaled_pts[k] = (pt[0]*self.scale[0],pt[1]*self.scale[1])

    return resized_map,footprint,scaled_pts
