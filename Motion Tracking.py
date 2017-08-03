import numpy as np
import cv2
from cv2 import bgsegm


#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('people-walking.mp4')
#cap = cv2.VideoCapture('Ball_Bouncing.mp4')
#cap = cv2.VideoCapture('Walking.mp4')
cap = cv2.VideoCapture('Running.mp4')


Background_subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while (1):
    #ret and frame, first and next frame
    ret,frame = cap.read()
    first_frame = Background_subtract.apply(frame)
    gblur = cv2.GaussianBlur(first_frame, (5, 5), 0)


#contouring
    #threshold : Separate out regions of an image corresponding to objects which we want to analyze.
    ret, threshold = cv2.threshold(gblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    _, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print("number of contours detected", len(contours))

    cv2.drawContours(frame, contours, 0, (0, 0, 255), 6)

    cv2.imshow('original', frame)
    cv2.imshow('first frame', gblur)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    cv2.imshow('original', frame)
    cv2.imshow('first frame', gblur)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
