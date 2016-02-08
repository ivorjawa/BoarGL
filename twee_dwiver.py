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
import colorsys

import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import PIL, ImageFont, ImageDraw
import cv2, cv

from transforms import perspective, translate, rotate, zrotate, yrotate, xrotate

import scrunlib
import ubble
import trans_gl

import hid_js

import fft_subp

import cgrid
import ctree
import cbulb


# opencl stuff
from vector import Vec
#OpenCL code
import part2
#functions for initial values of particles
import initialize

def load_text(filename):
    with open(filename) as f:
        data = f.read()
        return data
    return None


ver_shade = load_text("twee.vsh")
frag_shade = load_text("twee.fsh")

#cap = cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)

view = None
projection = None


camera = None
FOV = 45
win_width = 1000
win_height = 1000

timestamp = time.time()


class Camera(scrunlib.Orientable):
    def __init__(self):
        super(Camera, self).__init__()

shader_data_block = None


# openCL instance
gCLE = None


lights_in = []
lights_out = []
lights_init = False
def xmas_get_lights():
    #
    global lights_in, lights_out, lights_init
    if lights_init == False:
        lights_init = True
        narf = [ [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0] ]
        lights_in = narf * 15
        lights_out = narf * 15
    l = lights_out.pop(0)
    lights_in.append(l)
    l = lights_in.pop(0)
    lights_out.append(l)
    return lights_out

def spectrum_get_lights():
    global lights_in, lights_out, lights_init
    if lights_init == False:
        lights_init == True
        hues = np.array(range(0, 360, 30)) * 1/360.0
        for hue in hues:
            vals = [.25, .5, .75, .90, 1.0, 1.0, 1.0, .90, .75, .5, .25]
            for val in vals:
                lights_in.append(list(colorsys.hsv_to_rgb(hue, 1, val)) + [1])
            #for i in range(5):
            #    lights_in.append([0, 0, 0, 1])
        lights_out = lights_in[:30]
    #hughes = [random() for i in range(30)]
    #lights = [list(colorsys.hsv_to_rgb(hue, 1, 1)) + [1]  for hue in hughes]
    l = lights_out.pop(0)
    lights_in.append(l)
    l = lights_in.pop(0)
    lights_out.append(l)
    return lights_out

trawler = None
def fft_get_lights():
    global trawler
    if trawler == None:
        trawler = fft_subp.Trawler()
    pix = None
    while(1): # empty queue, display latest
        p = trawler.fish()
        if p == None:
            break
        else:
            pix = p

    if pix == None:
        pix = spectrum_get_lights()
    pix = pix[:30]
    pix = [list(p) + [1.0] for p in pix]
    #print "\n\n%s\n\n" % str(pix)
    return pix

def get_lights():
    retval = fft_get_lights()
    #retval = spectrum_get_lights()
    return retval

def display():
    try:
        r_display()
    except Exception, e:
        print "shit ourselves in display."
        traceback.print_exc( )
        sys.exit(1)

cnt = 0

def r_display():
    global drawables, gCLE
    global shader_data_block

    global timestamp
    global camera, FOV

    global cnt

    now = time.time()
    timedelt = now-timestamp
    timestamp = now
    fps = 1.0/timedelt

    #ret, im = cap.read()
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    #im = np.zeros((480,640,3), np.uint8)
    #cam_texture.set_data(im)

    hid_js.read_js(camera)
    view = camera.matrix()

    shader_data_block.map()
    shader_data_block.set('u_view', view)
    shader_data_block.set("camera_position", np.array(camera.loc, dtype=np.float32))
    cnt += 1
    if cnt == 2:
        cnt = 0
        lights = get_lights()
        #print lights
        shader_data_block.set("light_diffuse", np.array(lights, dtype=np.float32))
    shader_data_block.unmap()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


    for drawable in drawables:
        drawable.draw()

    gCLE.draw()
    # render fountain



    glut.glutSwapBuffers()

def reshape(width,height):
    global FOV, win_width, win_height, shader_data_block
    win_width = width
    win_height = height
    gl.glViewport(0, 0, width, height)
    projection = perspective( FOV, width/float(height), 1.0, 1000.0 )

    shader_data_block.map()
    shader_data_block.set('u_projection', projection)
    shader_data_block.unmap()


def keyboard(key, x, y):
    hid_js.set_key(key)
    if key == '\033': sys.exit( )



def timer(fps):
    global theta, phi, drawables

    # we may still want to update per-drawable variables on their own
    for drawable in (drawables + [gCLE]):
        rotate(drawable.modvec, 1, 0, 0, 1)
        drawable.program.unif['u_model'] = drawable.modvec
        pass
    glut.glutTimerFunc(1000//fps, timer, fps)
    glut.glutPostRedisplay()

class CLOO(object):
    def __init__(self, vershade, fragshade):
        self.vershade = vershade
        self.fragshade = fragshade
        self.program = trans_gl.ProgBundle(self.vershade, self.fragshade)

        self.program.tex['flake_tex'] = "flake.png"
        rmat = np.eye(4)
        self.modvec = rmat
        self.program.unif['u_model'] = self.modvec

        #set up initial conditions
        self.num = 90000
        self.dt = .001
        (self.pos_vbo, self.col_vbo, self.vel, self.vao2) = initialize.fountain(self.num)
        #create our OpenCL instance
        self.cle = part2.Part2(self.num, self.dt)
        self.cle.loadData(self.pos_vbo, self.col_vbo, self.vel, self.vao2)
    def draw(self):
        self.cle.execute(10)
        gl.glFlush()
        gl.glUseProgram(self.program.program)
        self.program.tex.bind()
        self.cle.render()
        gl.glUseProgram(0)



def run():
    global view, projection, camera
    global drawables
    global shader_data_block
    global gCLE

    # Glut init
    # --------------------------------------
    hid_js.init_js()
    scrunlib.init_gl(sys.argv)

    light_pos = (5,20,-10, 0)
    #light = sphere.Light(*light_pos[:2], scale = 1)

    camera = Camera()

    #greenGrid = cgrid.Grid(ver_shade, frag_shade)
    twee = ctree.Tree(ver_shade, frag_shade)

    drawables = [
        #greenGrid,
        twee,
    ]
    bvsh = load_text("bulb.vsh")
    bfsh = load_text("bulb.fsh")
    for i, light in enumerate(twee.lights):
        drawables.append(cbulb.Bulb(bvsh, bfsh, i, twee.lights))

    # print some debug info and set up window and callbacks
    print gl.glGetString(gl.GL_RENDERER)
    print gl.glGetString(gl.GL_VERSION)
    #glut.glutReshapeWindow(1024,1024)
    glut.glutReshapeFunc(reshape)
    glut.glutKeyboardFunc(keyboard )
    glut.glutDisplayFunc(display)
    glut.glutTimerFunc(1000//30, timer, 30)

    # Build view/camera & projection ... model uniform in individual programs
    # --------------------------------------
    projection = np.eye(4,dtype=np.float32) # will be set in reshape()

    camera.pitch(m.pi/2)
    camera.trans_z(-6)
    camera.trans_y(2)

    view = camera.matrix()
    print "camera matrix:\n%s" % str(view)


    # set up our uniform data block, stuff shared between all programs
    shader_data_block = ubble.UBTalker("shader_data")

    camera_position = ubble.UBVar("camera_position", np.array(camera.loc, dtype=np.float32))
    main_light_position = ubble.UBVar("main_light_position", np.array(light_pos))
    main_light_diffuse = ubble.UBVar("main_light_diffuse", np.array([1, 1, 1, 1]))
    u_view = ubble.UBVar("u_view", view)
    u_projection = ubble.UBVar("u_projection", projection)

    nlights = len(twee.lights)

    #hughes = [i/nlights for i in range(nlights)]
    #lights = [list(colorsys.hsv_to_rgb(hue, 1, 1)) + [1]  for hue in hughes]
    lights = get_lights()

    #lights  = [[0, 0, 0, 1]] * 30
    #lights[-2] = [1, 0, 0, 1]
    #lights[-1] = [0, 0, 1, 1]

    #print "lights: %s" % str(lights)

    light_positions = ubble.UBArrayVar("light_positions", np.array( twee.lights, dtype=np.float32))
    light_diffuse = ubble.UBArrayVar("light_diffuse", np.array(lights, dtype=np.float32))

    vars = [camera_position, main_light_position, main_light_diffuse,
            u_view, u_projection, light_positions, light_diffuse]
    for var in vars:
        shader_data_block.addvar(var)

    # compile it?
    trans_gl.get_struct_ub(shader_data_block)

    print "uniform block stuff:"
    shader_data_block.dump()
    #print shader_data_block.packed()


    #print "drawables: %s" % str(drawables)
    for drawable in drawables:
        drawable.program.ublock["shader_data"] = shader_data_block

    # set up OpenCL instance
    gCLE = CLOO(load_text("snow.vsh"), load_text("snow.fsh"))
    gCLE.program.ublock["shader_data"] = shader_data_block

    # Start
    # --------------------------------------
    print "starting gl main loop ... "
    glut.glutMainLoop()
    print "returning from gl main loop"

if __name__ == "__main__":
    run()
