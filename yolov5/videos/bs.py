from __future__ import print_function
import cv2 as cv
import numpy as np
from skimage.measure import LineModelND, ransac
import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(history=50, dist2Threshold=300)
else:
    backSub = cv.createBackgroundSubtractorKNN(history=50, dist2Threshold=300)
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)

    # apply image dilation
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel2)
    fgMask = cv.medianBlur(fgMask,5)
    #_, fgMask = cv.threshold(fgMask, 127, 255, 0)
    

    # contours
    #contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #fgMask = cv.cvtColor(fgMask, cv.COLOR_GRAY2BGR)
    #fgMask = cv.drawContours(fgMask, contours, -1, (0,255,0), 3)

     
    
    #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    #cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
