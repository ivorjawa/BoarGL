#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import sys
import traceback
import time
import math as m
from random import random
import copy
import atexit
import os

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut


import BoarGL.baseutil
import BoarGL.gluniform
import BoarGL.glbase

class Camera(BoarGL.baseutil.Orientable):
    def __init__(self):
        super(Camera, self).__init__()

class BoarDriver(object):
    def run(self):
        print "starting gl main loop ... "
        glut.glutMainLoop()
    def __init__(self, fps=30):
        self.drawables = []
        self.vars = []
        self.projMat = np.eye(4,dtype=np.float32)
        self.camera = Camera()
        self.FOV = 45
        self.win_width = 1000
        self.win_height = 1000
        self.viewMat = self.camera.matrix()
        self.shader_data_block = BoarGL.gluniform.UniformBlock("shader_data")
        self.debug = False
        self.target_fps = fps
        self.current_fps = 0
        if self.debug:
            print gl.glGetString(gl.GL_RENDERER)
            print gl.glGetString(gl.GL_VERSION)
        glut.glutReshapeFunc(self._reshape)
        glut.glutKeyboardFunc(self._keyboard )
        glut.glutDisplayFunc(self._display)
        glut.glutTimerFunc(1000//self.target_fps, self._timer, self.target_fps)
        self.init_vars_drawables()
        self.set_datablock()
    def init_vars_drawables(self):
        raise NotImplementedError, "Attempt to instantiate BoarDriver"
    def timer(self):
        raise NotImplementedError, "Attempt to instantiate BoarDriver"
    def _timer(self, tfps):
        self.timer(tfps)
        glut.glutTimerFunc(1000//tfps, self._timer, tfps)
        glut.glutPostRedisplay()
    def keyboard(self, key, x, y):
        raise NotImplementedError, "Attempt to instantiate BoarDriver"
    def _keyboard(self, key, x, y):
        self.keyboard(key, x, y)
    def _display(self):
            try:
                self.display()
            except Exception, e:
                print "died in display."
                traceback.print_exc( )
                sys.exit(1)
    def display(self):
        raise NotImplementedError, "Attempt to instantiate BoarDriver"
    def _reshape(self, width, height):
        self.reshape(width, height)
    def reshape(self, width, height):
        raise NotImplementedError, "Attempt to instantiate BoarDriver"
    def set_datablock(self):
        for var in self.vars:
            self.shader_data_block.addvar(var)

        # set up binding point index
        BoarGL.glbase.pack_copy(self.shader_data_block)

        if self.debug:
            print "uniform block stuff:"
            self.shader_data_block.dump()
            #print shader_data_block.packed()

        for drawable in self.drawables:
            drawable.program.ublock["shader_data"] = self.shader_data_block
