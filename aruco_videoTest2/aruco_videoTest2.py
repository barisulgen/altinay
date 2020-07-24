# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 03:30:21 2020

@author: baris
"""

import numpy as np
import cv2 as cv
from cv2 import aruco
import matplotlib as mpl
import matplotlib.pyplot as plt

test_mtx = np.loadtxt("test_mtx.txt", delimiter = ',')
test_dist = np.loadtxt("test_dist.txt", delimiter = ',')

# specified the aruco marker library that will be used
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# created parameters for the detectMarkers process
parameters =  aruco.DetectorParameters_create() 

# started the video capturing
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera...")
    exit()
while True:
    # camera resolution settings
    ret = cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    ret = cam.set(cv.CAP_PROP_FRAME_HEIGHT, 960)
    # Capture frame-by-frame
    ret, frame = cam.read() # returns the image and "true" value if grabbed
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive a frame...")
        break
    
    # TIMESTAMP_TEST
    #cv.putText(frame, "deneme", (10,700), 1, 1.1, (255, 255, 255))

    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters = parameters)
    markers = aruco.drawDetectedMarkers(frame, corners, ids, borderColor = (0,255,0))
    cv.imshow('markers', markers)
    
    # CAMERA_CALIBRATION_TEST
    #reto = aruco.estimatePoseSingleMarkers(corners, 8, test_mtx, test_dist)
    #rvec, tvec = reto[2][0,0,:], reto[2][1,0,:]
    #aruco.drawDetectedMarkers(frame, corners)
    #aruco.drawAxis(frame, test_mtx, test_dist, rvec, tvec, 8)
    #cv.imshow("frame", frame)
    
    
    # for 20 fps, waitKey is set to 0.05 seconds/50 miliseconds 
    # press "q" to quit
    if cv.waitKey(50) == ord('q'):
        break
    
cam.release()
cv.destroyAllWindows()

