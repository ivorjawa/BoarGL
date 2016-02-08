from __future__ import division

import os
import numpy as np
import math as m
import OpenGL.GL as gl


import BoarGL.glbase
import BoarGL.gldraw
import BoarGL.geom.cube

A = np.array

from BoarGL.glbase import d2r

# theta [spherical -> cartesian], lambda (mercator) -> longitude, x, u
# phi -> latitude, y, v

def sph_cart(lon, lat, scale = 1):
    "spherical to cartesian mapping"
    lon = d2r(lon)
    lat = d2r(lat)

    rsl = m.cos(lat)
    y = m.sin(lat)
    x = rsl*m.cos(lon)
    z = rsl*m.sin(lon)

    return np.array([scale*x, scale*y, scale*z,  1])

def spherical(lon, lat):
    #lat = latfn(lat)
    u = 1 - ((lon / 360) + .5)
    v = 1 - ((lat / 180) + .5)
    return (u, v)

def sphere_geom(tx=0, ty=0, tz=0, scale=.25):
    step = 15
    lons = range(-180, 180, step)+[180]
    lats = range(-90, 90, step)

    trans_v = np.array([tx, ty, tz, 0])

    # debugging stuff
    #self.lons = lons
    #self.lats = lats
    #self.pointpairs = []

    verts = []
    uv = []
    norms = []
    inds = []
    cols = []
    cols2 = []
    stash = []

    offset = 0
    rpt = False
    #print "lats: %s" % str(lats)
    for (lnum, lat) in enumerate(lats):
        #if lat == lats[0]:
        #    #print "bottom row"
        #if lat == lats[-1]:
        #    #print "top row"
        #print "entering lat: %s" % lat
        for lon in lons:
            p0 = (lon, lat)
            p1 = (lon, lat+step)

            if (lon == lons[0]): # stash first point
                stash = [offset+0, offset+1]

            points = [p0, p1]
            #self.pointpairs.append(points)
            verts = verts + [(sph_cart(p[0], p[1], scale)+trans_v) for p in points]
            norms = norms + [sph_cart(p[0], p[1], scale)[:3] for p in points]
            #uv = uv + [mercator(*p) for p in points]
            uv = uv + [spherical(*p) for p in points]
            #b = bands_9_15[lnum]
            #cols = cols + [b for p in points]
            cols = cols + [[1, 1, 1, 1] for p in points]


            if rpt: # build a degenerate triangle on boundaries
                inds.append(offset)
                rpt = False
            inds = inds + [offset + 0, offset + 1]
            offset = offset + 2

        # at end of a circle, produce a degenerate triangle
        if(lat != lats[-1]):
            inds.append(stash[1])
            rpt = True

    verts = BoarGL.gldraw.flar(verts)# * scale
    norms = BoarGL.gldraw.flar(norms)
    cols = np.array(cols)
    uv = np.array(uv)
    #print "verts: ", verts.shape
    #print "norms: ", norms.shape
    #print "cols: ", cols.shape
    #print "uvs: ", uv.shape
    vb1 = BoarGL.gldraw.VertexBundle(verts, norms, cols, uv)
    return vb1

class Bulb(object):
    def __init__(self, vershade, fragshade, bulb_index, bulb_coords):
        self.vershade = vershade
        self.fragshade = fragshade
        self.program = BoarGL.glbase.ProgBundle(self.vershade, self.fragshade)



        #vertices, filled_ind, outline_ind = cube.cube(.05)
        #print "I have %d vertices" % len(vertices)
        #print "%d indices" % len(filled_ind)
        #print "index: %d, bulb cooard: %s" % (bulb_index, str(bulb_coords[bulb_index]))
        #self.num_inds = len(filled_ind)
        #self.vao = boargl_base.copyBuffer(vertices, filled_ind)

        bci = bulb_coords[bulb_index]
        self.vb = sphere_geom(bci[0], bci[1], bci[2], .03)
        self.vb.bufferize()
        self.vao = BoarGL.glbase.copyBuffer(self.vb.cooked_faces, self.vb.cooked_ind)

        rmat = np.eye(4)
        #translate(rmat, bci[0], bci[1], bci[2])
        #trans.rotate(rmat, 90, 1, 0, 0) # rotate forward 90 degrees on x axis
        #trans.translate(rmat, 0, .0001, 0)
        self.modvec = rmat
        self.program.unif['u_model'] = self.modvec
        self.program.unif['bulb_index'] = bulb_index

    def draw(self):
        gl.glUseProgram(self.program.program)
        gl.glBindVertexArray(self.vao)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        nf = len(self.vb.cooked_ind)
        #print "about to draw %s faces" % nf
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, nf)
        gl.glBindVertexArray(0)
