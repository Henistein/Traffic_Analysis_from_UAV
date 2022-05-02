import math
import numpy as np
import cv2

# camera intrinsics
W = 1280
H = 720
fov_angle = 84*math.pi/180
F = int((W/2) / math.tan(fov_angle/2))
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
Kinv = np.linalg.inv(K)

real_points = np.array([(0,0,32),(0,7.6,32),(7.6,0,32),(0,-7.6,32),(-7.6,0,32)])
image_points = np.array([(642,416),(642,316),(750,416),(642,516),(534,416)], dtype=np.float32)

"""
for i in range(1, 5):
  wX = real_points[i,0]
  wY = real_points[i,1]
  wd = real_points[i,2]

  d1 = np.sqrt(np.square(wX)+np.square(wY))
  wZ = np.sqrt(np.square(wd)-np.square(d1))
  real_points[i,2] = wZ

"""

success, rvec, tvec = cv2.solvePnP(real_points,image_points,K,None,flags=0)

R, jac = cv2.Rodrigues(rvec)

np.set_printoptions(suppress=True)
print(tvec)
print(R)
exit(0)



add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

cap = cv2.VideoCapture('rotunda2.MP4')
while(cap.isOpened()):
  ret, frame = cap.read()
  if not ret:
    print('No frames grabbed!')
    break
  
  #frame = np.einsum('ij,klj->kli',np.linalg.inv(K),frame)

  #3x3 @ 3x1 = 3x1

  cv2.imwrite('rotunda.jpg', frame)
  exit(0)
  #if cv2.waitKey(30) & 0xff == 27:
  #  break
cv2.destroyAllWindows()
