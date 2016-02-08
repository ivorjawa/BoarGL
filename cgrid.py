from __future__ import division

import os
import numpy as np
A = np.array
#import math as m
import OpenGL.GL as gl
#import OpenGL.GLUT as glut

from transforms import translate, rotate

import trans_gl  #opengl utilities, all the various advanced opengl block stuff.  covers some same ground as scrunlib
import trans_draw # higher level convenience objects, program and vertext bundles
import cube

class Grid(object):
    def __init__(self, vershade, fragshade):
        self.vershade = vershade
        self.fragshade = fragshade
        self.program = trans_gl.ProgBundle(self.vershade, self.fragshade)


        #steps = 10 # keep it even to have (0,0) on a line
        #uvr = np.arange(0, 1.1, 1/steps) # 0 .. 1
        #xyr = (uvr - .5) * steps # 1 per unit, +- steps/2 wide

        #verts = []
        #tex_coord = []
        # verticals first
        #for x in range(steps+1):
        #    verts.append( A([xyr[x], xyr[0], 0]) )
        #    tex_coord.append( A([uvr[x], uvr[0]]) )
        #    verts.append( A([xyr[x], xyr[-1], 0]) )
        #    tex_coord.append( A([uvr[x], uvr[-1]]) )

        # horizontals first
        #for y in range(steps+1):
        #    verts.append( A([xyr[0], xyr[y], 0]) )
        #    tex_coord.append( A([uvr[0], uvr[y]]) )
        #    verts.append( A([xyr[-1], xyr[y], 0]) )
        #    tex_coord.append( A([uvr[-1], uvr[y]]) )

        #verts = A(verts, dtype=np.float32)
        #norms = A([ [0, 0, 1] for v in verts ], dtype=np.float32)
        #cols = A([ [1, 0, 0, 1] for v in verts], dtype=np.float32)
        #uvs = tex_coord

        #self.vb = trans_draw.VertexBundle(verts, norms, cols, uvs)
        #self.vb.bufferize() # does v, f, o = scrunlib.geomBuffer(self)
        #self.vao = trans_gl.copyBuffer(self.vb.cooked_faces, self.vb.cooked_ind)

        vertices, filled_ind, outline_ind = cube.cube(3)
        print "I have %d vertices" % len(vertices)
        print "%d indices" % len(filled_ind)
        self.num_inds = len(filled_ind)
        self.vao = trans_gl.copyBuffer(vertices, filled_ind)


        rmat = np.eye(4)
        translate(rmat, 0, 2, 0)
        #trans.rotate(rmat, 90, 1, 0, 0) # rotate forward 90 degrees on x axis
        #trans.translate(rmat, 0, .0001, 0)
        self.modvec = rmat
        self.program.unif['u_model'] = self.modvec

    def draw(self):
        gl.glUseProgram(self.program.program)
        gl.glBindVertexArray(self.vao)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        nf = self.num_inds
        #print "about to draw %s faces" % nf
        gl.glDrawArrays(gl.GL_LINES, 0, nf)
        gl.glBindVertexArray(0)
