from OpenGL.GL import *


import timing
timings = timing.Timing()

@timings
def fountain_np(num):
    """numpy way of initializing data using ufuncs instead of loops"""
    import numpy
    pos = numpy.ndarray((num, 4), dtype=numpy.float32)
    col = numpy.ndarray((num, 4), dtype=numpy.float32)
    vel = numpy.ndarray((num, 4), dtype=numpy.float32)

    rad = 10.001
    r2 = 1.
    pos[:,0] = numpy.sin(numpy.arange(0., num) * rad * numpy.pi / num)
    pos[:,0] *= numpy.random.random_sample((num,)) / r2 + .2
    pos[:,1] = numpy.cos(numpy.arange(0., num) * rad * numpy.pi / num)
    pos[:,1] *= numpy.random.random_sample((num,)) / r2 + .2
    pos[:,2] = 4.5  # z coordinate
    pos[:,3] = 1.

    col[:,0] = 0.
    col[:,1] = 1.
    col[:,2] = 0.
    col[:,3] = .5

    vel[:,0] = pos[:,0] * 2.
    vel[:,1] = pos[:,1] * 2.
    vel[:,2] = 1. # z velocity
    vel[:,3] = 1.8*numpy.random.random_sample((num, )) # life

    return pos, col, vel

@timings
def fountain_loopy(num):
    """This is a slower way of initializing the points (by 10x for large num)
    but more illustrative of whats going on"""

    from math import sqrt, sin, cos
    import numpy
    pos = numpy.ndarray((num, 4), dtype=numpy.float32)
    col = numpy.ndarray((num, 4), dtype=numpy.float32)
    vel = numpy.ndarray((num, 4), dtype=numpy.float32)

    import random
    random.seed()
    for i in xrange(0, num):
        rad = random.uniform(1, 3);
        x = sin(2*3.14 * i/num)*rad
        z = 0.
        y = cos(2*3.14 * i/num)*rad

        pos[i,0] = x
        pos[i,1] = y
        pos[i,2] = z
        pos[i,3] = 1.

        col[i,0] = 0.
        col[i,1] = 1.
        col[i,2] = 0.
        col[i,3] = 1.

        life = random.random()
        vel[i,0] = x*2.
        vel[i,1] = y*2.
        vel[i,2] = 3.
        vel[i,3] = life

    return pos, col, vel

def fountain(num):
    """Initialize position, color and velocity arrays we also make Vertex
    Buffer Objects for the position and color arrays"""
    import numpy

    #pos, col, vel = fountain_loopy(num)
    pos, col, vel = fountain_np(num)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    print timings

    #create the Vertex Buffer Objects
    from OpenGL.arrays import vbo
    pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    pos_vbo.bind()
    col_vbo = vbo.VBO(data=col, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    col_vbo.bind()

    #pos_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo.buffer)
    #glBufferData(GL_ARRAY_BUFFER, 4*len(pos), pos,
    #             GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)

    tc = numpy.ndarray((num, 2), dtype=numpy.float32)
    tc_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, tc_vbo)
    glBufferData(GL_ARRAY_BUFFER, 2*len(tc), tc,
                 GL_STATIC_DRAW)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)

    nm = numpy.ndarray((num, 3), dtype=numpy.float32)
    nm_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, nm_vbo)
    glBufferData(GL_ARRAY_BUFFER, 3*len(nm), nm,
                 GL_STATIC_DRAW)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)

    #col_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo.buffer)
    #glBufferData(GL_ARRAY_BUFFER, 4*len(col), col,
    #             GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(3)
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, None)

    glBindVertexArray(0)

    return (pos_vbo, col_vbo, vel, vao)


