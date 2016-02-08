#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import sys
import time
import math as m
import colorsys

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut

from BoarGL.driver import BoarDriver
import BoarGL.baseutil
import BoarGL.gluniform
import BoarGL.glbase
from BoarGL.baseutil import load_text
from BoarGL.gluniform import UBVar, UBArrayVar

from BoarGL.ext.transforms import perspective, rotate

import hid_js
import fft_subp
import spiral_tree
import cbulb
from snow_globe import SnowGlobe

class SpecTree(BoarDriver):
    def __init__(self):
        self.light_count = 0
        self.timestamp = time.time()
        self.lights_in = []
        self.lights_out = []
        self.lights_init = False
        self.trawler = None
        if "-xmas" in sys.argv:
            self.get_lights = self.xmas_get_lights
        elif "-spectrum" in sys.argv:
            self.get_lights = self.spectrum_get_lights
        else:
            self.get_lights = self.fft_get_lights
        BoarDriver.__init__(self)
    def init_vars_drawables(self): # called by super.init()
        tree = spiral_tree.Tree(load_text("shaders/tree.vsh"),
                                load_text("shaders/tree.fsh"))
        self.drawables.append(tree)
        bvsh = load_text("shaders/bulb.vsh")
        bfsh = load_text("shaders/bulb.fsh")
        for i, light in enumerate(tree.lights):
            self.drawables.append(cbulb.Bulb(bvsh, bfsh, i, tree.lights))

        snowglobe = SnowGlobe(load_text("shaders/snow.vsh"),
                              load_text("shaders/snow.fsh"))
        self.drawables.append(snowglobe)

        self.projection = np.eye(4,dtype=np.float32)
        # will be modified in reshape()

        self.camera.pitch(m.pi/2)
        self.camera.trans_z(-6)
        self.camera.trans_y(2)

        self.viewMat = self.camera.matrix()

        camera_position = UBVar("camera_position",
                                np.array(self.camera.loc, dtype=np.float32))
        light_pos = (5,20,-10, 0)
        main_light_position = UBVar("main_light_position",
                                    np.array(light_pos))
        main_light_diffuse = UBVar("main_light_diffuse",
                                   np.array([1, 1, 1, 1]))
        u_view = UBVar("u_view", self.viewMat)
        u_projection = UBVar("u_projection", self.projection)

        nlights = len(tree.lights)

        lights = self.get_lights()


        light_positions = UBArrayVar("light_positions",
                                     np.array( tree.lights,
                                               dtype=np.float32))
        light_diffuse = UBArrayVar("light_diffuse",
                                   np.array(lights, dtype=np.float32))

        self.vars = [camera_position, main_light_position,
                     main_light_diffuse,u_view, u_projection,
                     light_positions, light_diffuse]

    def timer(self, tfps):
        # we can update uniforms in each of the drawables separately
        # see also display()
        for drawable in self.drawables:
            rotate(drawable.modvec, 1, 0, 0, 1)
            drawable.program.unif['u_model'] = drawable.modvec
    def keyboard(self, key, x, y):
        hid_js.set_key(key)
        if key == '\033': sys.exit( )
    def display(self):
        now = time.time()
        timedelt = now-self.timestamp
        self.timestamp = now
        self.current_fps = 1.0/timedelt

        hid_js.read_js(self.camera)
        view = self.camera.matrix()

        # updating shared variables in shader data block.  Also see
        # timer()
        self.shader_data_block.map()
        self.shader_data_block.set('u_view', view)
        self.shader_data_block.set("camera_position",
                                   np.array(self.camera.loc,
                                            dtype=np.float32))
        self.light_count += 1
        if self.light_count == 2:
            self.light_count = 0
            self.lights = self.get_lights()
            self.shader_data_block.set("light_diffuse",
                                  np.array(self.lights, dtype=np.float32))
        self.shader_data_block.unmap()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


        for drawable in self.drawables:
            drawable.draw()

        glut.glutSwapBuffers()
    def get_lights(self):
        retval = self.fft_get_lights()
        #retval = self.spectrum_get_lights()
        return retval
    def reshape(self, width, height):
        self.win_width = width
        self.win_height = height
        gl.glViewport(0, 0, width, height)
        self.projection = perspective( self.FOV,
                                       width/float(height), 1.0, 1000.0 )
        self.shader_data_block.map()
        self.shader_data_block.set('u_projection', self.projection)
        self.shader_data_block.unmap()
    def cycle_lights(self):
        l = self.lights_out.pop(0)
        self.lights_in.append(l)
        l = self.lights_in.pop(0)
        self.lights_out.append(l)
        return self.lights_out
    def xmas_get_lights(self):
        if self.lights_init == False:
            self.lights_init = True
            bulbpat = [ [1, 0, 0, 1],
                        [0, 1, 0, 1],
                        [0, 0, 1, 1],
                        [1, 1, 0, 0] ]
            self.lights_in = bulbpat * 15
            self.lights_out = bulbpat * 15
        return self.cycle_lights()
    def spectrum_get_lights(self):
        if self.lights_init == False:
            self.lights_init == True
            hues = np.array(range(0, 360, 30)) * 1/360.0
            for hue in hues:
                vals = [.25, .5, .75, .90, 1.0, 1.0, 1.0, .90, .75, .5, .25]
                for val in vals:
                    self.lights_in.append(list(colorsys.hsv_to_rgb(hue, 1, val)) + [1])
            self.lights_out = self.lights_in[:30]
        return self.cycle_lights()
    def fft_get_lights(self):
        if self.trawler == None:
            self.trawler = fft_subp.Trawler()
        pix = None
        while(1): # empty queue, display latest
            p = self.trawler.fish()
            if p == None:
                break
            else:
                pix = p

        if pix == None:
            pix = self.spectrum_get_lights()
        pix = pix[:30]
        pix = [list(p) + [1.0] for p in pix]
        return pix

def run():
    hid_js.init_js()
    BoarGL.glbase.init_gl(sys.argv)
    print gl.glGetString(gl.GL_RENDERER)
    print gl.glGetString(gl.GL_VERSION)
    driver = SpecTree()
    driver.run()

if __name__ == "__main__":
    run()
