#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/12/14

@author: Sammy Pfeiffer

From http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration

Extra info:
http://docs.opencv.org/2.4.10/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=calibratecamera#cv2.calibrateCamera

I love this guy "OpenCV error messages suck":
https://adventuresandwhathaveyou.wordpress.com/2014/03/14/opencv-error-messages-suck/

Get pose:
http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_pose/py_pose.html


"""
import numpy as np
import cv2
import glob
import math
import pickle

def _pdist(p1, p2):
    """
    Distance bwt two points. p1 = (x, y), p2 = (x, y)
    """
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

print "Initializing"
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

n_rows = 5
n_cols = 4
n_cols_and_rows = (n_cols, n_rows) #originally (7,6) # 4,5 same results
n_rows_and_cols = (n_rows, n_cols)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((n_rows*n_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

mypath = "/home/sampfeiffer/svn/argus_ws/src/argus_tools/data/calibrationdata_camera_1/"
print "Getting images from " + mypath
images = glob.glob(mypath + '*.png')
#print "images is: " + str(images)

criteria_calibrator = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1 )
criteria = criteria_calibrator

for idx, fname in enumerate(images):
    print "\nImage " + fname
#     if idx > 10:
#         break
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, n_rows_and_cols,None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        print "  found " + str(len(corners)) + " corners."
        objpoints.append(objp)
        # cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) didnt work, I couldnt make it work copying the calibrator code
        imgpoints.append(corners)

        # Draw and display the corners
        #cv2.drawChessboardCorners(img, n_rows_and_cols, corners, ret)

        #cv2.imshow('img',img)
        #cv2.waitKey(500)

print "objpoints len: " + str(len(objpoints))
print "imgpoints len: " + str(len(imgpoints))


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#inf = open("./calibration_return_values.pickle", "rb" )
inf = open("./calibration_return_values_rows_and_cols.pickle", "rb" )

datathings = pickle.load(inf)
ret, mtx, dist, rvecs, tvecs = datathings

#datathings = (ret, mtx, dist, rvecs, tvecs)

# fieldnames = ["ret", "mtx", "dist", "rvecs", "tvecs"]
# for fieldname, data in zip(fieldnames, datathings):
#     print fieldname + ": "
#     print data

tot_error = 0.0
mean_error = 0.0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    print "current error: " + str(error)
    mean_error += error

print "total error: ", mean_error/len(objpoints)
# This should be as close to zero as possible.
# total error:  0.125571679917


#cv2.destroyAllWindows()
