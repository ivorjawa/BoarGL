#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import traceback
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import cv2, cv
import time
import math as m

from BoarGL.ext.transformations import quaternion_about_axis, quaternion_matrix


def load_text(filename):
    with open(filename) as f:
        data = f.read()
        return data
    return None

class Orientable(object):
    def __str__(self):
        return "Orientable\nloc: %s\nroll: %s\npitch: %s\nyaw: %s\n" % (
            self.loc, self.roll_axis, self.pitch_axis, self.yaw_axis)
    def __init__(self):
        self.loc = np.array([0, 0, 0, 0])
        self.up = np.array([0, 1, 0, 0]) #immutable up axis
        self.roll_axis = np.array([0, 0, 1, 0]) # rotate along z for roll
        self.yaw_axis = np.array([0, 1, 0, 0]) # rotate along y for yaw
        self.pitch_axis = np.array([1, 0, 0, 0]) # rotate long x for pitch
    def matrix(self):
        at = self.loc + self.roll_axis
        #up = self.up
        up = self.yaw_axis # awesome, this works but I need to be able to control my roll axis
        mat = CalLookAtMatrix(self.loc, at, up).T
        return mat
    def roll(self, angle):
        q = quaternion_about_axis(angle, self.roll_axis)
        rmat = quaternion_matrix(q).T
        self.yaw_axis = (np.dot(self.yaw_axis, rmat))
        self.pitch_axis = (np.dot(self.pitch_axis, rmat))
    def pitch(self, angle):
        q = quaternion_about_axis(angle, self.pitch_axis)
        rmat = quaternion_matrix(q).T
        self.yaw_axis = (np.dot(self.yaw_axis, rmat))
        self.roll_axis = (np.dot(self.roll_axis, rmat))
    def yaw(self, angle):
        q = quaternion_about_axis(angle, self.yaw_axis)
        rmat = quaternion_matrix(q).T
        self.roll_axis = (np.dot(self.roll_axis, rmat))
        self.pitch_axis = (np.dot(self.pitch_axis, rmat))
    def trans_x(self, dist):
        delta = dist * self.pitch_axis
        self.loc = self.loc + delta
        #print self
    def trans_y(self, dist):
        delta = dist * self.yaw_axis
        self.loc = self.loc + delta
        #print self
    def trans_z(self, dist):
        delta = dist * self.roll_axis
        self.loc = self.loc + delta
        #print self

def rounded_rectangle(src, topLeft, bottomRight, lineColor,
                      thickness=1, lineType=cv.CV_AA , cornerRadius=10):
    #/* corners:
    # * p1 - p2
    # * |     |
    # * p4 - p3
    # */
    p1 = topLeft;
    p2 = (bottomRight[0], topLeft[1]);
    p3 = bottomRight;
    p4 = (topLeft[0], bottomRight[1]);

    #// draw straight lines
    cv2.line(src, (p1[0]+cornerRadius,p1[1]),
         (p2[0]-cornerRadius,p2[1]),
         lineColor, thickness, lineType);
    cv2.line(src,(p2[0],p2[1]+cornerRadius),
         (p3[0],p3[1]-cornerRadius),
         lineColor, thickness, lineType);
    cv2.line(src, (p4[0]+cornerRadius,p4[1]),
         (p3[0]-cornerRadius,p3[1]),
         lineColor, thickness, lineType);
    cv2.line(src, (p1[0],p1[1]+cornerRadius),
         (p4[0],p4[1]-cornerRadius),
         lineColor, thickness, lineType);

    #// draw arcs
    cv2.ellipse( src, tuple(np.add(p1, (cornerRadius, cornerRadius))),
                 ( cornerRadius, cornerRadius ), 180, 0, 90,
                 lineColor, thickness, lineType );
    cv2.ellipse( src, tuple(np.add(p2, (-cornerRadius, cornerRadius))),
                 ( cornerRadius, cornerRadius ), 270, 0, 90,
                 lineColor, thickness, lineType );
    cv2.ellipse( src, tuple(np.add(p3, (-cornerRadius, -cornerRadius))),
                 ( cornerRadius, cornerRadius ), 0, 0, 90,
                 lineColor, thickness, lineType );
    cv2.ellipse( src, tuple(np.add(p4, (cornerRadius, -cornerRadius))),
                 ( cornerRadius, cornerRadius ), 90, 0, 90,
                 lineColor, thickness, lineType );
def plot_normals(faces):
    furtle = faces.copy()
    nurtle = faces.copy()
    z = np.vstack((furtle,nurtle)).reshape((-1,),order='F')
    for i in range(len(faces)):
        base = z[2*i]
        end = z[2*i + 1]
        offset = base[0]
        normal = end[2]
        soem = np.add(offset, normal*.25)
        # I obviously don't understand something about numpy
        z[2*i + 1][0][0] = soem[0]
        z[2*i + 1][0][1] = soem[1]
        z[2*i + 1][0][2] = soem[2]
    return z

def norm_color(vexor, offset, col):
    #again, glaring defficiency here
    vexor[offset][3][0] = col[0]
    vexor[offset][3][1] = col[1]
    vexor[offset][3][2] = col[2]
    vexor[offset][3][3] = col[3]


""" indexing
http://wiki.scipy.org/Tentative_NumPy_Tutorial

In [23]: A = np.array([ [1,2,3], [4,5,6], [7,8,9] ])

In [24]: A
Out[24]:
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

# first row
In [44]: A[0,:]
Out[44]: array([1, 2, 3])

# first column
In [45]: A[:, 0]
Out[45]: array([1, 4, 7])

In [57]: np.append([1, 2, 3], 4)
Out[57]: array([1, 2, 3, 4])

"""

# this generates the worldToCamera matrix
nrm = lambda a: a / np.linalg.norm(a)
def CalLookAtMatrix(cameraPt, lookPt, upPt):
    cameraPt = np.array(cameraPt)[:3]
    lookPt = np.array(lookPt)[:3]
    upPt = np.array(upPt)[:3]

    lookDir = nrm(lookPt - cameraPt)

    upDir = nrm(upPt)
    rightDir = np.cross(lookDir, upDir)
    perpUpDir = np.cross(rightDir, lookDir)

    rotMat = np.eye(4)

    rotMat[:,0] = np.append(rightDir, 0)
    rotMat[:,1] = np.append(perpUpDir, 0)
    rotMat[:,2] = np.append(lookDir, 0)*-1

    rotMat = rotMat.T # is this even needed?
    transMat = np.eye(4)
    transMat[:,3] = np.append(cameraPt*-1, 1)
    retval = np.dot(rotMat, transMat)
    retval = np.array(retval, dtype=np.float32)
    return retval

def testLookAt():
    cameraPt = [5, 6, 5]
    lookPt = [0, 0, 0]
    upPt = [1, 10, 1]

    return CalLookAtMatrix(cameraPt, lookPt, upPt)

