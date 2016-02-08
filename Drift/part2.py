#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush
from OpenGL.GL import *
from OpenGL.GLU import *


import pyopencl as cl
import sys, os

import numpy
import Drift.timing

timings = Drift.timing.Timing()
class Part2(object):
    def __init__(self, num, dt, *args, **kwargs):
        self.clinit()
        self.loadProgram("Drift/part2.cl");

        self.num = num
        self.dt = numpy.float32(dt)

        self.timings = timings



    def loadData(self, pos_vbo, col_vbo, vel, vao):
        import pyopencl as cl
        mf = cl.mem_flags
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo

        self.pos = pos_vbo.data
        self.col = col_vbo.data
        self.vel = vel
        self.vao = vao

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        #self.pos_vbo.bind()
        #For some there is no single buffer but an array of buffers
        #https://github.com/enjalot/adventures_in_opencl/commit/61bfd373478767249fe8a3aa77e7e36b22d453c4
        try:
            self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffer))
            self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffer))
        except AttributeError:
            self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))
            self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))
        self.col_vbo.bind()

        #pure OpenCL arrays
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
        self.pos_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pos)
        self.vel_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vel)
        self.queue.finish()

        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.pos_cl, self.col_cl]



    @timings
    def execute(self, sub_intervals):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)

        global_size = (self.num,)
        local_size = None

        kernelargs = (self.pos_cl,
                      self.col_cl,
                      self.vel_cl,
                      self.pos_gen_cl,
                      self.vel_gen_cl,
                      self.dt)

        for i in xrange(0, sub_intervals):
            self.program.part2(self.queue, global_size, local_size, *(kernelargs))

        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
        self.queue.finish()


    def clinit(self):
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        import sys
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
            print "cl context: %s" % self.ctx
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)

        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print fstr
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def render(self):

        #glEnable(GL_POINT_SMOOTH)
        glPointSize(6)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        #self.col_vbo.bind()
        #glColorPointer(4, GL_FLOAT, 0, self.col_vbo)

        #self.pos_vbo.bind()
        #glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)

        #glEnableClientState(GL_VERTEX_ARRAY)
        #glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.num)
        glBindVertexArray(0)

        #glDisableClientState(GL_COLOR_ARRAY)
        #glDisableClientState(GL_VERTEX_ARRAY)

        #glDisable(GL_BLEND)


