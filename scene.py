from __future__ import division # / is always float, // is integer division

import math
import numpy
from numpy import array

import OpenGL.GL as gl
import OpenGL.GLUT as glut

from vispy.gloo import Program, VertexBuffer, IndexBuffer, Texture2D

import scrunlib

GL_POINTS            =             0x0000
GL_LINES             =             0x0001
GL_LINE_LOOP         =             0x0002
GL_LINE_STRIP        =             0x0003
GL_TRIANGLES         =             0x0004
GL_TRIANGLE_STRIP    =             0x0005
GL_TRIANGLE_FAN      =             0x0006
GL_QUADS             =             0x0007
GL_QUAD_STRIP        =             0x0008
GL_POLYGON           =             0x0009

class PVObject(object):
  def __init__(self, val):
    self.val = val

class PVFloat(PVObject): pass
class PVInt(PVObject): pass
class PVString(PVObject): pass
class PVVec3(PVObject): pass
class PVVec4(PVObject): pass
class PVMat(PVObject): pass

# float fzNear = 1.0f; float fzFar = 100.0f;
# [cameraToClipMatrix setVec:0 x:fFrustumScale];
# [cameraToClipMatrix setVec:1 y:fFrustumScale];
# [cameraToClipMatrix setVec:2 z:(fzFar + fzNear) / (fzNear - fzFar)];
# [cameraToClipMatrix setVec:2 w:-1.0f];
# [cameraToClipMatrix setVec:3 z:(2 * fzFar * fzNear) / (fzNear - fzFar)];

def calcFrustrumScale(fovDeg):
  rads = (math.pi * fovDeg) / 180.0
  retval = 1.0/math.tan(rads/2.0)
  return retval


class Scene(object):
  def __init__(self, vs, fs, model=None):
    self.vershade = scrunlib.load_text(vs)
    self.fragshade = scrunlib.load_text(fs)
    self.program = Program(self.vershade, self.fragshade)
    self.model = model
    self.frustrumScale = 1
    self.cameraToClipMatrix = numpy.eye(4)
    self.zFar = 0
    self.zNear = 0
    self.fovDeg = 0
    self.stream = False
    self.streamMat = False
    self.primmode = GL_TRIANGLES
    self.batchsize = 0
    self.set_offset([0, 0, 0])
    self.rootNode = None
  def set_pers(self, zNear, zFar, fovDeg):
    self.zFar = zFar
    self.zNear = zNear
    self.fovDeg = fovDeg
    # this generates the clip matrix / projection / cameraToClipMatrix
    self.frustrumScale = calcFrustrumScale(fovDeg)
    fscale = self.frustrumScale
    pers =  array([[fscale, 0, 0, 0],
                     [0, fscale, 0, 0],
                     [0, 0, (zFar + zNear) / (zNear - zFar), -1],
                     [0,0,(2 * zFar * zNear) / (zNear - zFar), 0]]).T
    self.cameraToClipMatrix = pers
  def __repr__(self):
      return """Scene <%s>
Vertex Shader: %s
Fragment Shader: %s
Model: %s
Stream: %s
Stream Matrix: %s
Root Node: %s""" % (id(self), self.vershade, self.fragshade, self.model, self.stream, self.streamMat, self.rootNode)

  def load(self):
    # get the geometry loaded
    print "Scene::load"
    return self.model
  def update(self):
    # update the geometry for the next frame -- single model
    print "Scene::reload"
    return self.model
  def animate(self, vars):
    # reloads matrices and updates geometry for nodal object models
    # yes, this is weak.
    print "Scene::animate"
    print self
    return {}
  def get_key_mat(self, key, modMat):
    print "Scene::get_key_mat key: %s \nmat: %s" % (key, modMat);
    return modMat
  def set_offset(self, offset):
    self.offset = array(offset)
