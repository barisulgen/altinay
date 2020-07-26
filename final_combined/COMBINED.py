# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:08:02 2020

@author: baris
"""
#%%
from cv2 import aruco
import numpy as np
import cv2 as cv
import glob
import math

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((1,(4,4)[0]*(4,4)[1], 3),np.float32)
objp[0,:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('clb_pics_tel_resize/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #(tot, gray) = cv.threshold(gray2, 128, 255, cv.THRESH_BINARY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (4,4), cv.CALIB_CB_ADAPTIVE_THRESH)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, (4,4), corners2,ret)
        cv.imshow('img',img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


#mtx = np.loadtxt("mtx.txt", delimiter = ',')
#dist = np.loadtxt("dist.txt", delimiter = ',')




#%%

#ROTATIONS, from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create() 
font = cv.FONT_HERSHEY_PLAIN

cam = cv.VideoCapture('vid_rot.mp4')
#out = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('X','V','I','D'), 30, (1696,960))

#cam.set(cv.CAP_PROP_FRAME_WIDTH, 1696)
#cam.set(cv.CAP_PROP_FRAME_HEIGHT, 960)

#if not cam.isOpened():
    #print("Cannot open camera...")
    #exit()
check = True
while True:
    ret, frame = cam.read() # returns the image and "true" value if grabbed
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive a frame...")
        break
    timeStamp = str(int(cam.get(cv.CAP_PROP_POS_MSEC)/1000))
    cv.putText(frame, "Elapsed Time(sec): "+timeStamp, (10, 15), font,  0.6, (50, 50, 250), 1, cv.LINE_4)
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(image = img_gray, dictionary = aruco_dict, parameters = parameters, cameraMatrix = mtx, distCoeff = dist)
    if ids is not None and ids[0] == 13:
        ret = aruco.estimatePoseSingleMarkers(corners, 8, mtx, dist)
        rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
        
        aruco.drawDetectedMarkers(frame, corners, ids, borderColor = (150,0,200))
        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 8)

        #-- Center indicator
        frame = cv.line(frame,(416,240),(432,240),(50,50,200),1)
        frame = cv.line(frame,(424,232),(424,248),(50,50,200),1)
        frame = cv.circle(frame,(424,240), 4, (50,50,250), 1)
        
        #-- Print the tag position in camera frame
        str_position = "Marker Position(centimeters): x=%2.0f  y=%2.0f  z=%2.0f"%(tvec[0], tvec[1], tvec[2])
        cv.putText(frame, str_position, (10, 55), font, 0.8, (180, 50, 0), 1, cv.LINE_AA)

        #-- Obtain the rotation matrix tag->camera
        R_ct    = np.matrix(cv.Rodrigues(rvec)[0])
        R_tc    = R_ct.T

        #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)
        
        #-- Print the marker's attitude respect to camera frame
        str_attitude = "Marker Attitude(degrees): r=%2.0f  p=%2.0f  y=%2.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),math.degrees(yaw_marker))
        cv.putText(frame, str_attitude, (10, 70), font, 0.8, (180, 50, 0), 1, cv.LINE_AA)

    
    frame = cv.resize(frame,(1696,960),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
    cv.imshow('frame', frame)
    #out.write(frame)
    
    if check == True:
        print(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        print(cam.get(cv.CAP_PROP_FRAME_WIDTH))
        check = False
    
    key = cv.waitKey(20) & 0xFF
    if key == ord('q'):
        cam.release()
        cv.destroyAllWindows()
        break

