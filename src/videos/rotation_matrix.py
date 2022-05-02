import numpy as np

# https://cameratransform.readthedocs.io/en/latest/spatial.html

def cos(x):
  return np.cos(np.pi*x/180)

def sin(x):
  return np.sin(np.pi*x/180)

#  the rotation of the image. (0°: camera image is not rotated (landscape format), 90°: camera image is in portrait format, 180°: camera is in upside down landscape format)
aroll = 0

# the tilt of the camera. (0°: camera faces down, 90°: camera faces parallel to the ground, 180°: camera faces upwards)
atilt = 0

# the direction in which the camera is looking. (0°: the camera faces “north”, 90°: east, 180°: south, 270°: west)
aheading = 0

#Rx = np.array([1,0,0],[0,cos(angleX),-sin(angleX)],[0,sin(angleX),cos(angleX)])
#Ry = np.array([[cos(angleY),0,sin(angleY)],[0,1,0],[-sin(angleY),0,cos(angleY)]])
#Rz = np.array([[cos(angleZ),-sin(angleZ),0],[sin(angleZ),cos(angleZ),0],[0,0,1]])

Rroll = np.array([[cos(aroll),sin(aroll),0],[-sin(aroll),cos(aroll),0],[0,0,1]])
Rtilt = np.array([[1,0,0],[0,cos(atilt),sin(atilt)],[0,-sin(atilt),cos(atilt)]])
Rheading = np.array([[cos(aheading),-sin(aheading),0],[sin(aheading),cos(aheading),0],[0,0,1]])

R = Rroll @ Rtilt @ Rheading
print(R)
