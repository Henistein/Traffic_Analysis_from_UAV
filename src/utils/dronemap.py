import math
import numpy as np
import cv2
import pandas as pd
import geopy

from osgeo import osr, ogr, gdal
from .heatmap import HeatMap

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
  
  def pixels_to_coords(self, pixel, line):
    ul_x = self.geo_matrix[0]
    ul_y = self.geo_matrix[3]
    x_dist = self.geo_matrix[1]
    y_dist = self.geo_matrix[5]
    x = pixel*x_dist + ul_x
    y = line*y_dist + ul_y
    return x, y

class GeoInterpolation:
  """
  geo = GeoInterpolation(
      lat_north=40.273668,
      long_west=-7.506679,
      lat_south=40.268085,
      long_east=-7.493513,
      lat_center=40.271018,
      long_center=-7.500335,
      image=cv2.imread('images/MAPA.jpg')
  )
  """
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
  # DJI Phantom 4 Pro V2 parameters
  # https://docs.google.com/spreadsheets/d/1w5PXRstTtZ0xMOewYdNrtzpvyVXHISPNwa65XuqcXEI/edit#gid=0
  SENSOR_WIDTH = 13.2 # mm
  FOCAL_LENGTH = 8.8 # mm
  IMAGE_WIDTH = 5477
  IMAGE_HEIGHT = 3612
  # frame
  FRAME_WIDTH = 1280
  FRAME_HEIGHT = 720
  # resized map 
  RESIZED_MAP_WIDTH = 1280
  RESIZED_MAP_HEIGHT = 610
  def __init__(self, drone_data, geo, n_frames):
    self.geo = geo
    self.map = self.geo.image
    # extract drone data
    data = pd.read_csv(drone_data)
    max_data = int(n_frames//3)
    self.latitude_list = data["latitude"][:max_data].tolist()
    self.longitude_list = data["longitude"][:max_data].tolist()
    self.compass_heading_list = data[" compass_heading(degrees)"][:max_data].tolist()
    self.height_list = data["height_above_takeoff(feet)"][:max_data].tolist()
    self.max_data = len(self.latitude_list)
    # feet to meters
    self.height_list = [h*0.3048 for h in self.height_list]
    # ground sample distance
    self.gsd = None
    self.FOOTPRINT_WIDTH = None
    self.FOOTPRINT_HEIGHT = None
    # heatmap
    self.heatmap = HeatMap(self.map)
  
  def _update_footprint_values(self, height):
    """
    Update the footprint values
    :param height: drone height
    """
    self.gsd = (self.SENSOR_WIDTH*height)*100/(self.FOCAL_LENGTH*self.IMAGE_WIDTH)
    self.FOOTPRINT_WIDTH = (self.gsd*self.IMAGE_WIDTH)/100
    self.FOOTPRINT_HEIGHT = (self.gsd*self.IMAGE_HEIGHT)/100

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

    # draw footprint on the map
    cv2.drawContours(img,[box],0,(0,0,255),thickness=10)
    
    footprint,_,_ = self._crop_rect(self.map.copy(),rect)

    # rotate image to match drone perspective
    if angle >= 270 or angle == 0:
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif angle >= 180:
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle >= 90:
      footprint = cv2.rotate(footprint, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img,footprint

  def scale(self, x_meters):
    # scale(X meters) = Y pixels
    # 100 pixels
    lat1, lon1 = self.geo.pixels_to_coords(0, 0)
    lat2, lon2 = self.geo.pixels_to_coords(0, 100)
    # calculate distance in meters
    dist = geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).meters
    return 100*x_meters/dist

  def get_data(self, k):
    """
    Get data from the list
    :param k: index of the data
    @return: latitude, longitude, compass_heading
    """
    return self.latitude_list[k], self.longitude_list[k], \
           self.compass_heading_list[k], self.height_list[k]

  def get_next_data(self, k, idcenters, frame, detections):
    """
    Process the data from the drone
    Extract the drone footprint from the map
    Draw the points from the drone frame to the map
    :param k: index of the drone data list
    :param idcenters: list frame object centers
    """
    mapp = self.map.copy()
    # get data from logs and store last data
    lat,lon,heading,height = self.get_data(k)

    # convert drone current position to map coordinates
    x,y = self.geo.coords_to_pixels(lat,lon)

    # draw xy on the map
    mapp = cv2.circle(mapp,(x,y),radius=3,color=(0,0,255),thickness=10)

    # update drone footprint
    #height = 60
    height = 125
    #height = 94
    self._update_footprint_values(height)

    # convert footprint to pixels
    ftp_X_px, ftp_Y_px = self.scale(self.FOOTPRINT_WIDTH/2), self.scale(self.FOOTPRINT_HEIGHT/2)

    # draw footprint on the map
    mapp,footprint = self.draw_drone_footprint((x,y),heading,mapp,ftp_X_px,ftp_Y_px)

    """
    Convert video frame points to map coordinates
    - Scale video frame points to cropped footprint
    """
    # scale used to map video frame points to cropped footprint
    ftp_scale_x = MapDrone.FRAME_WIDTH/(ftp_X_px*2)
    ftp_scale_y = MapDrone.FRAME_HEIGHT/(ftp_Y_px*2)
    # auxiliar variables to convert from footprint to map coordinates
    incx, incy = x-ftp_X_px, y-ftp_Y_px

    scaled_pts = {}
    for k,pt in idcenters.items():
      # scaling the video frame points to cropped footprint 
      pt = (pt[0]/ftp_scale_x, pt[1]/ftp_scale_y)
      # convert from cropped footprint to map coordinates
      pt = (pt[0]+incx, pt[1]+incy)
      # rotate points to match drone heading
      pt = self.rotate_camera((x,y), heading, [pt])[0]
      # update heatmap points
      self.heatmap.update_points(pt)
      # draw point on map
      mapp = cv2.circle(mapp,(int(pt[0]),int(pt[1])),radius=3,color=(0,255,0),thickness=10)
      # convert points to lat and lon
      scaled_pts[k] = self.geo.pixels_to_coords(pt[0],pt[1])

    # resize mapp
    mapp = cv2.resize(mapp,(MapDrone.RESIZED_MAP_WIDTH,MapDrone.FRAME_HEIGHT))
    return mapp,footprint,scaled_pts

