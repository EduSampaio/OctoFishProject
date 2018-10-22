#! /usr/bin/env python

"""
Copyright 2018 Fritz Alexander Francisco <fritz.a.francisco@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cv2
import os, sys
import glob
import random
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# input arguments
ap = argparse.ArgumentParser()
ap.add_argument('-r','--right_input',type=str,required=True, help="path to right video")
ap.add_argument('-l','--left_input',type=str,required=True, help="path to left video")
ap.add_argument('-v','--verbose',type=bool,default=False,help="visualize cameras")
ap.add_argument('-o','--output',type=str,default='.',help="output path")
ap.add_argument('-n','--num-images',type=int,default=20)
ap.add_argument('-c','--calib_pattern',nargs='+',type=int,default= [9,7],help='checkerboard pattern (rows, columns)')
args = vars(ap.parse_args())
print(args)

videos = (args['right_input'],args['left_input'])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
stereo_flags = cv2.CALIB_FIX_INTRINSIC

# define checkerboard pattern
# pattern_columns = 8
# pattern_rows = 6
pattern_columns = args['calib_pattern'][0]
pattern_rows = args['calib_pattern'][1]

# Filenames are just an increasing number
frameId = 0
counter = 0

frame_shape = None
frame_list = []

# initiate data library
data = {}
for i, video in enumerate(videos):
    data[i] = {"ts":[],"Rs":[],"Ps":[],"Ks":[],"xh": [],"imgpts": [],"dst":[]}

right = cv2.VideoCapture(videos[0])
left = cv2.VideoCapture(videos[1])

objp = np.zeros((pattern_rows * pattern_columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_rows, 0:pattern_columns].T.reshape(-1, 2)
objpoints = []

ret_l, leftFrame = left.read()
ret_r, rightFrame = right.read()

# make sure both cameras captured frame:
while (ret_l == True) and (ret_r == True):
    print('Processing frame:',frameId,end='\r')
    # only views every tenth frame for a more random choice. True random image pick not yet implemented.
    if (counter < args['num_images']) and (frameId % 10 == 0):
        leftFrame = cv2.cvtColor(leftFrame,cv2.COLOR_BGR2GRAY)
        rightFrame = cv2.cvtColor(rightFrame,cv2.COLOR_BGR2GRAY)

        left_cal, corners_l = cv2.findChessboardCorners(leftFrame, (pattern_rows,pattern_columns), None,flags=find_chessboard_flags)
        right_cal, corners_r = cv2.findChessboardCorners(rightFrame, (pattern_rows,pattern_columns),None, flags=find_chessboard_flags)

        # synchronously find checkerboard
        if (left_cal == True) and (right_cal == True):
            corners_l = cv2.cornerSubPix(leftFrame,corners_l,(11,11),(-1,-1),criteria)
            corners_r = cv2.cornerSubPix(rightFrame,corners_r,(11,11),(-1,-1),criteria)

            if frame_shape == None:
                frame_shape = rightFrame.shape
            h,w = frame_shape[:2]

            for i,obj in enumerate(videos):
                if i == 0:
                    side = str('right')
                    data[i]['imgpts'].append(corners_r)

                else:
                    side = str('left')
                    data[i]['imgpts'].append(corners_l)

            objpoints.append(objp)
            frame_list.append(frameId)

            print('Found calibration image pair:',int(counter+1),'/',int(args['num_images']))
            counter += 1

    ret_l, leftFrame = left.read()
    ret_r, rightFrame = right.read()
    frameId += 1

    if args['verbose'] == True:
        leftFrame = cv2.resize(leftFrame,dsize=None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)
        rightFrame = cv2.resize(rightFrame,dsize=None,fx=0.3,fy=0.3,interpolation = cv2.INTER_LINEAR)

        cv2.imshow('left',leftFrame)
        cv2.waitKey(1)
        print("left:", ret_l)

        cv2.imshow('right',rightFrame)
        cv2.waitKey(1)
        print("right:", ret_r)

    # Stereo calibrate once all checkerboard images have been detected
    if counter == args['num_images']:
        cameraMatrix1 = None
        distCoeffs1 = None
        cameraMatrix2 = None
        distCoeffs2 = None
        R = None
        T = None
        E = None
        F = None
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints=objpoints,
                                                                                            imagePoints1=data[0]['imgpts'],
                                                                                            imagePoints2=data[1]['imgpts'],
                                                                                            imageSize=frame_shape,
                                                                                            distCoeffs1=distCoeffs1,
                                                                                            distCoeffs2=distCoeffs2,
                                                                                            cameraMatrix1=cameraMatrix1,
                                                                                            cameraMatrix2=cameraMatrix2,
                                                                                            criteria = stereo_criteria,
                                                                                            flags = stereo_flags
                                                                                            )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix1,
                                              cameraMatrix2=cameraMatrix2,
                                              distCoeffs1=distCoeffs1,
                                              distCoeffs2=distCoeffs2,
                                              imageSize=frame_shape,
                                              R=R,
                                              T=T,
                                              flags=cv2.CALIB_ZERO_DISPARITY,
                                              alpha= 1
                                              )

        right_maps = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1,frame_shape, cv2.CV_16SC2)
        left_maps = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, frame_shape, cv2.CV_16SC2)

        fig = plt.figure()
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111, projection='3d')

        for i,idx in enumerate(frame_list):
            # right.set(cv2.CAP_PROP_POS_FRAMES,idx)
            # left.set(cv2.CAP_PROP_POS_FRAMES,idx)
            #
            # r, right_frame  = right.read()
            # l, left_frame  = left.read()
            #
            # right_img_remap = cv2.remap(right_frame, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
            # left_img_remap = cv2.remap(left_frame, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
            #
            # # x1,y1,w1,h1 = roi1
            # # x2,y2,w2,h2 = roi2
            # # right_frame= right_frame[y1:y1+h1, x1:x1+w1]
            # # left_frame= left_frame[y2:y2+h2, x2:x2+w2]
            #
            # right_img_remap = cv2.resize(right_img_remap, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_CUBIC)
            # left_img_remap = cv2.resize(left_img_remap, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_CUBIC)
            # right_frame = cv2.resize(right_frame, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_CUBIC)
            # left_frame = cv2.resize(left_frame, None, fx = 0.3, fy = 0.3, interpolation = cv2.INTER_CUBIC)
            #
            # # See the results
            # view = np.hstack([right_frame, left_frame])
            # rectView = np.hstack([right_img_remap, left_img_remap])
            #
            # cv2.imshow('view', view)
            # cv2.imshow('rectView', rectView)
            # # Wait indefinitely for any keypress
            # cv2.waitKey(0)

            # Create 3D points through triangulation
            X = cv2.triangulatePoints( P1, P2, data[0]['imgpts'][i], data[1]['imgpts'][i])

            # Remember to divide out the 4th row. Make it homogeneous
            X /= X[3]
            X = X.T

            # # Recover the origin arrays from PX
            # x1 = np.dot(P1[:3],X)
            # x2 = np.dot(P2[:3],X)
            # # Again, put in homogeneous form before using them
            # x1 /= x1[2]
            # x2 /= x2[2]

            # plot with matplotlib
            Ys = X[:, 0]
            Zs = X[:, 1]
            Xs = X[:, 2]

            ax.scatter(Xs, Ys, Zs, marker='o')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_zlabel('X')
        plt.title('3D point cloud')
        plt.show()

        np.savez(os.path.join(args['output'] + "calibration_" + videos[0].replace(".mp4","_") + videos[1].replace(".mp4",".npz")), cameraMatrix1 = cameraMatrix1, distCoeffs1 = distCoeffs1, cameraMatrix2 = cameraMatrix2, distCoeffs2 = distCoeffs2, R = R,T = T, E = E, F = F)
        print("Successfully calibrated stereo camera!")
        break
