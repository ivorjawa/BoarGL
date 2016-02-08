from __future__ import division

import os
import numpy as np
A = np.array
import math as m
import OpenGL.GL as gl
import cv2, cv

import trans_gl, trans_draw

from trans_gl import sph_cart, d2r


bulb_inds = [1,111,168,211,247,278,307,333,357,379,401,
             421,440,459,476,494,510,526,542,557,572,586,
             600,614,628,641,654,666,679,691] # calculated offline

class Tree(object):
    def __init__(self, vershade, fragshade):
        print "starting tree init ..."
        self.vershade = vershade
        self.fragshade = fragshade
        self.program = trans_gl.ProgBundle(self.vershade, self.fragshade)
        nlights = 30
        self.lights = [[0, 0, 0, 1]] * 30
        mul = 10
        div = 1.0 / mul
        top = 360 * mul
        theta = np.array([d2r(t) for t in range(0, top, 5)+[top]])
        r = (div*theta) / (2*m.pi)
        r2 = (div*theta) / (2.2*m.pi)

        #inside and outside
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        height = 4
        z1 = height - (height * (theta/(2*m.pi*mul)))
        z2 = height - ((height - .2) * (theta/(2*m.pi*mul)))


        self.lights = zip(
            x[bulb_inds],
            y[bulb_inds],
            (z1[bulb_inds]+z2[bulb_inds])/2,
            [1 for i in bulb_inds])

        verts_f = []
        verts_b = []
        uvs = []
        norms_f = []
        norms_b = []
        cols_f = []
        cols_b = []

        col2 = [0, 1, 0, 1]
        col1 = [0, .5, 0, 1]
        j = 0
        #light_p = len(theta) // nlights
        for i, angle in enumerate(theta):

            # generate positions for nlights christmas lights
            #if i % light_p == 0:
            #    if(j < nlights):
            #        self.lights[j] = A([x[i], (y1[i]+y2[i])/2, z[i], 1])
             #       #print "setting light %d position: %s" % (j, str(self.lights[j]))
             #   j += 1
                #print "i: %d, j: %d nlights: %d, angles: %d" % (i, j, nlights, len(theta))

            p1 = A([x[i], y[i], z1[i], 1])
            p2 = A([x[i], y[i], z2[i], 1])

            fnorm = A([np.cos(angle), np.sin(angle), 0])
            verts_f = verts_f + [p1, p2]
            norms_f = norms_f + [-1*fnorm, -1*fnorm]
            verts_b = verts_b + [p2, p1]
            norms_b = norms_b + [1*fnorm, 1*fnorm]
            if(angle == 0):
                uvs = uvs + [(0, 0), (0, 1)]
            else:
                uvs = uvs + [(i/len(theta), 0), (i/len(theta), 1)]
            cols_f = cols_f + [col1, col1]
            cols_b = cols_b + [col2, col2]

        #print "about to bundle"
        vb1 = trans_draw.VertexBundle(verts_f, norms_f, cols_f, uvs)
        vb2 = trans_draw.VertexBundle(verts_b, norms_b, cols_b, uvs)
        #print "about to glue"
        self.vb = trans_draw.VertexBundle.glue_strips(vb1, vb2)
        #print "about to bufferize"
        self.vb.bufferize()

        #print "have %d faces, and %d indices." % (len(self.vb.cooked_faces), len(self.vb.cooked_ind))
        #print "about to copy buffer"
        self.vao = trans_gl.copyBuffer(self.vb.cooked_faces, self.vb.cooked_ind)
        #print "copied buffer"

        rmat = np.eye(4)
        self.modvec = rmat
        self.program.unif['u_model'] = self.modvec
        #print "ending tree init"

    def draw(self):
        #print "starting tree draw"
        gl.glUseProgram(self.program.program)
        gl.glBindVertexArray(self.vao)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.vb.cooked_ind))
        #self.program.draw(gl.GL_TRIANGLE_STRIP, self.vb.IndBuf)
        #self.norm_pb.program.draw(gl.GL_LINES, self.norm_vb.IndBuf)
        gl.glBindVertexArray(0)
        #print "ending tree draw"
