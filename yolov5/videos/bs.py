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
    backSub = cv.createBackgroundSubtractorMOG2(history=50)
else:
    backSub = cv.createBackgroundSubtractorKNN(history=250, dist2Threshold=250)
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

def histogram_equalization(img):
    # Perform Histogram Equalization to counter illumination change problem
    # convert the color scheme to YCrCb which separate the intensity/brightness information of image separately unlike rgb
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    
    # perform histogram equalization on the Y-channel of this image
    # for this, we use CLAGE's algorithm (that performs piecewise histogram equalization)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    img_ycrcb[:,:,0] = clahe.apply(img_ycrcb[:,:,0])
    
    # convert back to bgr
    img = cv.cvtColor(img_ycrcb, cv.COLOR_YCR_CB2BGR)
    return img

def adaptive_thresholding(img):
    channels = []
    for channel in cv.split(img):
        channel_bg = cv.medianBlur(cv.dilate(channel, np.ones((3,3))), 25)
        channel_edges = cv.absdiff(channel, channel_bg)
        channel_edges = cv.normalize(channel_edges, dst = None, norm_type = cv.NORM_MINMAX, alpha = 0, beta = 255)
        channels.append(channel_edges)
    return cv.merge(channels)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    

    #histogram_equalization(frame)
    adaptive_thresholding(frame)
    frame = backSub.apply(frame)

    # apply image dilation
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    #frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    #frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel2)
    #frame = cv.medianBlur(frame,5)

    # dynamic bgs
    frame = cv.erode(frame, kernel)
    frame = cv.erode(frame, kernel)
    frame = cv.dilate(frame, kernel)
    frame = cv.dilate(frame, kernel)

    frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel2)
    frame = cv.medianBlur(frame, 7)
    

    # contours
    #contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    #frame = cv.drawContours(frame, contours, -1, (0,255,0), 3)
    #print(contours)

     
    
    #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #           cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    #cv.imshow('Frame', frame)
    cv.imshow('FG Mask', frame)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
