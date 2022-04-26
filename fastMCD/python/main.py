import numpy as np
import cv2
import MCDWrapper
import sys

np.set_printoptions(precision=2, suppress=True)
cap = cv2.VideoCapture(sys.argv[1])
mcd = MCDWrapper.MCDWrapper()
isFirst = True
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    if (isFirst):
        mcd.init(gray)
        isFirst = False
    else:
        mask = mcd.run(gray)
    frame[mask > 0, 2] = 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
      area = cv2.contourArea(cnt)
      # threshold
      if area > 10:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0, 255, 0),3)
    #cv2.imshow('frame', frame)
    #cv2.imshow('frame', mask)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  
