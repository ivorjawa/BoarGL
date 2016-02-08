#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

import OpenGL.GL as gl

import BoarGL.glbase

from Drift.vector import Vec
import Drift.part2
import Drift.initialize

class SnowGlobe(object):
    def __init__(self, vershade, fragshade):
        self.vershade = vershade
        self.fragshade = fragshade
        self.program = BoarGL.glbase.ProgBundle(self.vershade, self.fragshade)

        self.program.tex['flake_tex'] = "flake.png"
        rmat = np.eye(4)
        self.modvec = rmat
        self.program.unif['u_model'] = self.modvec

        #set up initial conditions
        self.num = 90000
        self.dt = .001
        (self.pos_vbo, self.col_vbo, self.vel, self.vao2) = Drift.initialize.fountain(self.num)
        #create our OpenCL instance
        self.cle = Drift.part2.Part2(self.num, self.dt)
        self.cle.loadData(self.pos_vbo, self.col_vbo, self.vel, self.vao2)
    def draw(self):
        self.cle.execute(10)
        gl.glFlush()
        gl.glUseProgram(self.program.program)
        self.program.tex.bind()
        self.cle.render()
        gl.glUseProgram(0)
