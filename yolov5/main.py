import numpy as np
import cv2 as cv

from simple_inference import *

cap = cv.VideoCapture('cars.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (500, 500), interpolation = cv2.INTER_AREA)
    print(frame.shape)
    exit(0)
    frame = yolov5(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
