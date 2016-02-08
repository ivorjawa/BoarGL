#!/usr/bin/env python2.7
import numpy as np
import struct, array, ctypes

import OpenGL.GL as gl

floatsize = 4
intsize = 4

# this is some sort of variable translater system

class UBArrayVar(object):
    def __repr__(self):
        return "Name: %s, size: %d bytes, offset: %d bytes, eltsize: %d, value: |%s...|" % (self.name, self.size, self.offset,
            self.eltsize, str(self.value)[:20])
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.size = 0
        self.offset = 0
        t = type(value[0])
        if t == np.ndarray:
            self.eltsize = len(value[0])*floatsize
            self.fmt = 'f'*len(value[0])*len(value)
        else:
            self.eltsize = floatsize
            self.fmt = 'f'*len(value)
        self.size = self.eltsize * len(value)
        self.pak = struct.pack(self.fmt, *tuple(self.value.flatten()))

class UBVar(object):
    def __repr__(self):
        return "Name: %s, size: %d bytes, offset: %d bytes, value: |%s...|" % (self.name, self.size, self.offset, str(self.value)[:20])
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.size = 0
        self.offset = 0
        t = type(value)
        self.fmt = ''
        self.pak = ''
        if t == np.ndarray:
            if value.shape == (3,3):
                self.size = 3 * 3 * floatsize
                self.fmt = 'f'*9
            elif value.shape == (4, 4):
                self.size = 4 * 4 * floatsize
                self.fmt = 'f'*16
            elif value.shape == (3,):
                self.size = 3 * floatsize
                self.fmt = 'f'*3
            elif value.shape == (4,):
                self.size = 4 * floatsize
                self.fmt = 'f'*4
            else:
                raise RuntimeError, "I can't deal with %s: %s, %s" % (str(value), str(t), str(t.shape))
            self.pak = struct.pack(self.fmt, *tuple(self.value.flatten()))
        elif t == float:
            self.size = floatsize
            self.fmt = 'f'
            self.pak = struct.pack(self.fmt, self.value)
        elif t == int:
            self.size = intsize
            self.fmt = 'i'
            self.pak = struct.pack(self.fmt, self.value)
        else:
            raise RuntimeError, "I can't deal with %s: %s" % (str(value), str(t))

class UBTalker(object):
    def __init__(self, blockname):
        self.blockname = blockname
        self.lut = {}
        self.vars = []
        self.pointer = 0
        self.ubo = None # uniform block object
        self.bpi = None # binding point index
        self.vp = None
    def map(self):
        try:
            gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.ubo)
            self.vp = gl.glMapBuffer(gl.GL_UNIFORM_BUFFER, gl.GL_WRITE_ONLY)
        except Exception, e:
            print "ubo: %s" % str(self.ubo)
            raise
    def unmap(self):
        gl.glUnmapBuffer(gl.GL_UNIFORM_BUFFER)
        self.vp = None
    def packed(self):
        return ''.join([v.pak for v in self.vars])
    def addvar(self, ubvar):
        self.lut[ubvar.name] = len(self.vars)
        ubvar.offset = self.pointer
        self.pointer += ubvar.size
        self.vars.append(ubvar)
    def dump(self):
        print self.blockname, self.pointer
        for var in self.vars:
            print var
    def getvar(self, varname):
        return self.vars[self.lut[varname]]
    def locate(self, varname):
        var = self.getvar(varname)
        #print "UBTalker.locate called on %s with class %s" % (str(var), type(var))
        return var.offset, var.size
    def locate_elt(self, varname, index):
        var = self.vars[self.lut[varname]]
        return var.offset + index*var.eltsize, var.eltsize
    def set(self, varname, value):
        if self.vp == None:
            raise RuntimeError, "Map the buffer first, idiot"
        var = self.getvar(varname)
        if type(var) == UBArrayVar:
            z = np.fromstring(UBArrayVar(varname, value).pak, np.uint8)
        else:
            z = np.fromstring(UBVar(varname, value).pak, np.uint8)

        from_p = ctypes.c_void_p(z.ctypes.data)
        offset, size = self.locate(varname)
        to_p = ctypes.c_void_p(self.vp+offset)
        ctypes.memmove(to_p, from_p, size)
    def set_elt(self, varname, index, value):
        if self.vp == None:
            raise RuntimeError, "Map the buffer first, idiot"
        z = np.fromstring(UBVar(varname, value).pak, np.uint8)
        from_p = ctypes.c_void_p(z.ctypes.data)
        offset, size = self.locate_elt(varname, index)
        to_p = ctypes.c_void_p(self.vp+offset)
        ctypes.memmove(to_p, from_p, size)

def test():
    ubt = UBTalker("shader_data")
    u_view_var = UBVar("u_view", np.eye(4))
    u_blobby_var = UBArrayVar("u_blobby", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    u_lights_var = UBArrayVar("u_lights",
        np.array([ [ 1,  0,  0,  0,], [ 1,  0,  0,  0,], [ 1,  0,  0,  0,], [ 1,  0,  0,  0,], [ 1,  0,  0,  0,], ]))
    u_model_var = UBVar("u_model", np.eye(4))
    ubt.addvar(u_view_var)
    ubt.addvar(u_blobby_var)
    ubt.addvar(u_lights_var)
    ubt.addvar(u_model_var)

    ubt.dump()
    print "model", ubt.locate('u_model')
    print "blobby elt", ubt.locate_elt('u_blobby', 2)
    print "light elt", ubt.locate_elt('u_lights', 4)
    s = ubt.packed()
    print "packed len: %d" % len(s)
