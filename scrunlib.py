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

from cube import cube
#from vispy.gloo import Program, VertexBuffer, IndexBuffer, Texture2D
from transforms import perspective, translate, rotate
from transformations import quaternion_about_axis, quaternion_matrix

import pbox_util
import scene

# a scene has its own coordinate system, it's centered on one point
# a world contains many scenes, holds all common attributes

class World(object):
    def __init__(self, scenes):
        self.scenes = scenes
        self.cameraToClipMatrix = testLookAt()

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

def geomBuffer(g):
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal'  , np.float32, 3),
             ('a_color',    np.float32, 4)]


    itype = np.uint32
    p = g.verts
    n = g.norms #[g.norms[x] for x in range(0, 36, 6)]
    c = g.cols
    t = g.tex_coord[0]


    faces_p = g.inds.flatten()
    nv = len(faces_p)
    faces_c = faces_p
    faces_t = g.tex_coord_ind[0].flatten()
    faces_n = faces_p

    vertices = np.zeros(nv,vtype)
    try:
        vertices['a_position'] = p[faces_p]
        vertices['a_normal'] = n[faces_n]
        vertices['a_color'] = c[faces_c]
        vertices['a_texcoord'] = t[faces_t]
    except Exception, e:
        print "verts(p): %s" % str(p)
        print "inds: %s" % str(faces_p)
        print "normals(n) %s" % str(n)
        print "ni: %s" % str(faces_n)
        print "faces_c: %s" % str(faces_c)
        print "c: %s" % str(c)
        print "faces_t: %s" % str(faces_t)
        raise

    filled = np.array(range(nv), dtype=np.uint32)

    outline = np.resize( np.array([0,1,1,2,2,3,3,0], dtype=np.uint32), 6*(2*4))
    outline += np.repeat( 4*np.arange(6), 8)

    return vertices, filled, outline

class TexBox(object):
    def set_pers(self, zNear, zFar, fovDeg):
        self.zFar = zFar
        self.zNear = zNear
        self.fovDeg = fovDeg
        self.frustrumScale = scene.calcFrustrumScale(fovDeg)
        fscale = self.frustrumScale
        pers =  np.array([[fscale, 0, 0, 0],
                       [0, fscale, 0, 0],
                       [0, 0, (zFar + zNear) / (zNear - zFar), -1],
                       [0,0,(2 * zFar * zNear) / (zNear - zFar), 0]])
        self.cameraToClipMatrix = pers

    def __init__(self):
        self.scene = pbox_util.loadNodeScene(pbox_util.texcube_path)
        print "loaded %s" % pbox_util.texcube_path
        self.g = self.scene.geoms[0]
        verts, inds, fucknards = geomBuffer(self.g)
        self.verts = VertexBuffer(verts)
        self.inds = IndexBuffer(inds)
        self.vsh = load_text("texture.vsh")
        self.fsh = load_text("texture.fsh")
        self.program = Program(self.vsh, self.fsh)
        self.program.bind(self.verts)

        fzNear = 1
        fzFar = 1000.0
        fovDeg = 45
        self.set_pers(fzNear, fzFar, fovDeg)
        #self.cameraToClipMatrix =  np.eye(4,dtype=np.float32)
        self.program['cameraToClipMatrix'] = self.cameraToClipMatrix
        self.modelToWorldMatrix = np.eye(4,dtype=np.float32)
        self.program['modelToWorldMatrix'] = self.modelToWorldMatrix
        self.worldToCameraMatrix = np.eye(4,dtype=np.float32)
        self.program['worldToCameraMatrix'] = self.worldToCameraMatrix
        self.normalModelToWorldMatrix = np.eye(4,dtype=np.float32)
        self.program['normalModelToWorldMatrix'] = self.normalModelToWorldMatrix

        self.diffuse = self.g.material.diffuse[1]
        self.program['useDiffuseTexture'] = 1
        self.program['dirToLight'] = [2,2,2]
        self.program['lightIntensity'] = [1,1,1,1]
        self.texture = Texture2D(data = self.diffuse.image.getFloatArray(),
                                 format = 'rgb') # .pilimage.convert('RGBA'))
        self.program['diffuse_texture'] = self.texture

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

def init_gl(argv, win_name="Lighted Cube"):
    # Glut init
    # --------------------------------------
    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE |
                             glut.GLUT_RGBA |
                             glut.GLUT_DEPTH  |
                             glut.GLUT_3_2_CORE_PROFILE)


    #glut.glutCreateWindow(win_name)
    glut.glutGameModeString("2880x1800:32@60")
    # The application will enter fullscreen
    glut.glutEnterGameMode()

    print gl.glGetString(gl.GL_RENDERER)
    print gl.glGetString(gl.GL_VERSION)

    # OpenGL initalization
    # --------------------------------------
    # http://colorizer.org
    # np.array([38, 90, 120]) * 1/255.0
    #gl.glClearColor(.149,0.353,0.470,1) // nice grey blue
    #gl.glClearColor(2/255.0, 31/255.0, 131/255.0, 1) # teton blue
    gl.glClearColor(0, 0, 0, 1)

    # hide mouse cursor
    glut.glutSetCursor(glut.GLUT_CURSOR_NONE)

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    gl.glEnable(gl.GL_CULL_FACE);
    gl.glCullFace(gl.GL_BACK);
    gl.glFrontFace(gl.GL_CCW);

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthMask(gl.GL_TRUE);
    gl.glDepthFunc(gl.GL_LEQUAL);
    gl.glDepthRange(0.0, 1.0);

    gl.GL_TEXTURE_WRAP_S = gl.GL_CLAMP_TO_EDGE
    gl.GL_TEXTURE_WRAP_T = gl.GL_CLAMP_TO_EDGE

    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glLineWidth(0.75)

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


"""
    -(void) lookAtIt
{
    Vec3* lookPt = [Vec3 newx:LookX y:LookY z:LookZ];
    Vec3* upPt = [Vec3 newx:0 y:1 z:0];

    Vec3* camPos = ResolveCamPosition([self Phi],
                                      [self Theta],
                                      [self Radius],
                                      lookPt);
    lookMat = CalcLookAtMatrix(camPos, lookPt, upPt);
    NSLog(@"new LookAt matrix:");
    [lookMat log];
    for(id curScene in Scenes){
        [curScene setWorldMat:lookMat];
    }
}

Mat* CalcLookAtMatrix(Vec3* cameraPt, Vec3 *lookPt, Vec3 *upPt)
{

        //glm::vec3 lookDir = glm::normalize(lookPt - cameraPt);
    Vec3* lookDir = [lookPt sub:cameraPt];
    [lookDir normalize];
        //glm::vec3 upDir = glm::normalize(upPt);
    Vec3* upDir = [upPt mcopy];
    [upDir normalize];

        //glm::vec3 rightDir = glm::normalize(glm::cross(lookDir, upDir));
    Vec3* rightDir = [lookDir cross:upDir];
    [rightDir normalize];
        //glm::vec3 perpUpDir = glm::cross(rightDir, lookDir);
    Vec3* perpUpDir = [rightDir cross:lookDir];

        //glm::mat4 rotMat(1.0f);
    Mat* rotMat = [Mat newi];

    Vec4* comp = [Vec4 new];

        //rotMat[0] = glm::vec4(rightDir, 0.0f);
    [comp promote:rightDir W:0];
    [rotMat setCol:0 withVec:comp];

    //rotMat[1] = glm::vec4(perpUpDir, 0.0f);
    [comp promote:perpUpDir W:0];
    [rotMat setCol:1 withVec:comp];

    //rotMat[2] = glm::vec4(-lookDir, 0.0f);
    [comp promote:lookDir W:0];
    [comp scale:-1];
    [rotMat setCol:2 withVec:comp];


        //rotMat = glm::transpose(rotMat);
    [rotMat t];

        //glm::mat4 transMat(1.0f);
    Mat* transMat = [Mat newi];

        //transMat[3] = glm::vec4(-cameraPt, 1.0f);
    NSLog(@"cameraPoint:");
    [cameraPt log];
    [cameraPt scale:-1];
    NSLog(@"-cameraPoint:");
    [cameraPt log];
    [comp promote:cameraPt W:1];
    NSLog(@"comp:");
    [comp log];
    NSLog(@"transMat before");
    [transMat log];
    [transMat setCol:3 withVec:comp];

    NSLog(@"transMat after");
    [transMat log];

        //return rotMat * transMat;
    return [rotMat m_mul:transMat];
    //return [[rotMat m_mul:transMat] T];
}

    """
