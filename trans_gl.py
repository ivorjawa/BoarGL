#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import ctypes
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from PIL import Image
import math as m

import ubble

def init_gl(argv, win_name="Lighted Cube"):
    # Glut init
    # --------------------------------------
    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE |
                             glut.GLUT_RGBA |
                             glut.GLUT_DEPTH  |
                             glut.GLUT_3_2_CORE_PROFILE)
    glut.glutCreateWindow(win_name)
    print gl.glGetString(gl.GL_RENDERER)
    print gl.glGetString(gl.GL_VERSION)

    # OpenGL initalization
    # --------------------------------------
    # http://colorizer.org
    # np.array([38, 90, 120]) * 1/255.0
    gl.glClearColor(.149,0.353,0.470,1) #// nice grey blue
    #gl.glClearColor(2/255.0, 31/255.0, 131/255.0, 1) # teton blue
    #gl.glClearColor(.1, .1, .1, 1) # go full black when we have
    # walls and a roof and floor

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    gl.glEnable(gl.GL_CULL_FACE);
    gl.glCullFace(gl.GL_BACK);
    gl.glFrontFace(gl.GL_CW);

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthMask(gl.GL_TRUE);
    gl.glDepthFunc(gl.GL_LEQUAL);
    gl.glDepthRange(0.0, 1.0);

    gl.GL_TEXTURE_WRAP_S = gl.GL_CLAMP_TO_EDGE
    gl.GL_TEXTURE_WRAP_T = gl.GL_CLAMP_TO_EDGE

    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glLineWidth(0.75)

def compileShader(source, shaderType):
    """Compile shader source of given type
    source -- GLSL source-code for the shader
    shaderType -- GLenum GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc,
    returns GLuint compiled shader reference
    raises RuntimeError when a compilation failure occurs
    """
    if isinstance(source, str):
        print('string shader')
        source = [source]
    elif isinstance(source, bytes):
        print('bytes shader')
        source = [source.decode('utf-8')]

    shader = gl.glCreateShader(shaderType)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    result = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)

    if not(result):
        print "----------\n\n%s\n\n________\n" % gl.glGetShaderInfoLog( shader )
        # TODO: this will be wrong if the user has
        # disabled traditional unpacking array support.
        raise RuntimeError(
            """Shader compile failure (%s): %s"""%(
                result,
                gl.glGetShaderInfoLog( shader ),
                ),
            source,
            shaderType,
            )
    return shader

def loadShaders(strVS, strFS):
    """load vertex and fragment shaders from strings"""
    # compile vertex shader
    shaderV = compileShader([strVS], gl.GL_VERTEX_SHADER)
    # compiler fragment shader
    shaderF = compileShader([strFS], gl.GL_FRAGMENT_SHADER)

    # create the program object
    program = gl.glCreateProgram()
    if not program:
        raise RunTimeError('glCreateProgram faled!')

    # attach shaders
    gl.glAttachShader(program, shaderV)
    gl.glAttachShader(program, shaderF)

    # Link the program
    gl.glLinkProgram(program)

    # Check the link status
    linked = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not linked:
        infoLen = gl.glGetProgramiv(program, gl.GL_INFO_LOG_LENGTH)
        infoLog = ""
        if infoLen > 1:
            infoLog = gl.glGetProgramInfoLog(program, infoLen, None);
        gl.glDeleteProgram(program)
        raise RunTimeError("Error linking program:\n%s\n", infoLog);

    return program

def geomBuffer(g):
        vtype = [('a_position', np.float32, 4),
                 ('a_texcoord', np.float32, 2),
                 ('a_normal'  , np.float32, 3),
                 ('a_color',    np.float32, 4)]


        itype = np.uint32
        p = g.verts
        n = g.norms
        c = g.cols
        t = g.tex_coord[0]


        faces_p = g.inds.flatten()
        nv = len(faces_p)
        faces_c = faces_p
        faces_t = g.tex_coord_ind[0].flatten()
        faces_n = faces_p

        vertices = np.zeros(nv,vtype)
        try:
            vertices['a_position'] = p[faces_p]
            vertices['a_normal'] = n[faces_n]
            vertices['a_color'] = c[faces_c]
            vertices['a_texcoord'] = t[faces_t]
        except Exception, e:
            print "verts(p): %s" % str(p)
            print "inds: %s" % str(faces_p)
            print "normals(n) %s" % str(n)
            print "ni: %s" % str(faces_n)
            print "faces_c: %s" % str(faces_c)
            print "c: %s" % str(c)
            print "faces_t: %s" % str(faces_t)
            raise

        inds = np.array(range(nv), dtype=np.uint32)

        return vertices, inds

def copyBuffer(vertexData, indexData):
    "packs the output of a scrunlib.geomBuffer into a form that opengl can use"

    # set up vertex array object (VAO)
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    # vertices
    vertexBuffer = gl.glGenBuffers(1)
    indexPositions = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertexBuffer)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, indexPositions)


    # enable vertex array and set buffer data pointer
    # position
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 4*13, ctypes.c_void_p(0))

    # front texture coord
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4*13, ctypes.c_void_p(4*4))

    # normal
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 4*13, ctypes.c_void_p((4+2)*4))

    # color
    gl.glEnableVertexAttribArray(3)
    gl.glVertexAttribPointer(3, 4, gl.GL_FLOAT, gl.GL_FALSE, 4*13, ctypes.c_void_p((4+2+3)*4))

    # set buffer data
    gl.glBufferData(gl.GL_ARRAY_BUFFER, 13*len(vertexData)*4, vertexData,
                 gl.GL_STATIC_DRAW)
    indexData = np.array(indexData, dtype="uint16")
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, 2*len(indexData), indexData, gl.GL_STATIC_DRAW)


    # unbind VAO
    gl.glBindVertexArray(0)
    return vao

def loadTexture(filename):
    """load OpenGL 2D texture from given image file"""
    img = Image.open(filename)
    print "loaded image: %s with size: %s" % (filename, str(img.size))
    dur = list(img.getdata())
    #print dur
    imgData = np.array(dur, dtype=np.uint8)
    #print imgData
    texture = gl.glGenTextures(1)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
    #gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    #gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.size[0], img.size[1],
                 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, imgData)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture


class GLV(object): # Uh, GL Variable?  Base class for textures, uniforms, uniform blocks, as used in ProgBundles
    # used to provide .tex[], .unif[], and .ublock[] interface in program bundles
    def __init__(self, program):
        self.program = program
        self.lut = {}
    def __setitem__(self, index, value):
        gl.glUseProgram(self.program)
        v = self.classify(index, value)
        self.lut[index] = v

class VTex(GLV): # texture
    def __init__(self, program):
        super(VTex, self).__init__(program)
        self.tcx = {}
    def classify(self, name, obj):
        unif = gl.glGetUniformLocation(self.program, name)
        texture_id = loadTexture(obj) # filename
        self.tcx[unif] = texture_id
    def bind(self):
        for uniform, texture_id in self.tcx.iteritems():
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glUniform1i(uniform, 0)

signature = lambda a: (a.__class__ == np.ndarray) and (a.shape, a.__class__) or a.__class__
MAT4t = ((4,4), np.ndarray)
FLOAT3t = ((3,), np.ndarray)
FLOAT1t = (float)
INT1t = (int)
class VUnif(GLV): # uniform
    def __init__(self, program):
        super(VUnif, self).__init__(program)
    def classify(self, name, obj):
        sig = signature(obj)
        unif = gl.glGetUniformLocation(self.program, name)
        if sig == MAT4t:
            gl.glUniformMatrix4fv(unif, 1, gl.GL_FALSE, obj)
        elif sig == FLOAT1t:
            gl.glUniform1f(unif, obj)
        elif sig == FLOAT3t:
            gl.glUniform3fv(unif, 1, obj)
        elif sig == INT1t:
            gl.glUniform1i(unif, obj)
        else:
            raise RuntimeError, "Can't deal with this shit: %s, %s" % (str(obj), obj.__class__)
        return (unif, obj)

ub_binding_count = 0
def gen_ub():
    global ub_binding_count
    ubo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, ubo)
    binding_point_index = ub_binding_count
    ub_binding_count += 1
    return ubo, binding_point_index

def get_ub(listy, record_byte_size):
    ubo, binding_point_index = gen_ub()
    gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, binding_point_index, ubo)
    gl.glBufferData(gl.GL_UNIFORM_BUFFER, record_byte_size*len(listy), listy,
                        gl.GL_DYNAMIC_DRAW)
    gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, 0)
    return ubo, binding_point_index

def get_struct_ub(talker):
    ubo, binding_point_index = get_ub(talker.packed(), 1)
    talker.ubo = ubo
    talker.bpi = binding_point_index
    return ubo, binding_point_index

class UB(object):
    def __init__(self, record_byte_size, listy):
        self.record_byte_size = record_byte_size
        self.listy = listy
        self.ubo, self.bpidx = get_ub(listy, record_byte_size)
    def update(self, newlisty):
        self.listy = newlisty

class VUBlock(GLV): # uniform block
    def __init__(self, program):
        super(VUBlock, self).__init__(program)
        self.UBs = {}
    def classify(self, name, ub):
        #sig = signature(obj)
        uniform_idx = gl.glGetUniformBlockIndex(self.program, name)
        if(type(ub) == UB):
            ubo, bpidx = get_ub(ub.listy, ub.record_byte_size)
        else:
            ubo = ub.ubo
            bpidx = ub.bpi

        #FIXME
        # https://www.opengl.org/sdk/docs/man3/xhtml/glUniformBlockBinding.xml
        # http://stackoverflow.com/questions/23203095/whats-gluniformblockbinding-used-for
        # http://www.lighthouse3d.com/tutorials/glsl-core-tutorial/3490-2/

        #void glUniformBlockBinding(    GLuint program,
        #                               GLuint uniformBlockIndex,
        #                               GLuint uniformBlockBinding);


        # ubo, binding_point_index = get_ub(talker.packed(), 1)

        gl.glUniformBlockBinding(self.program, uniform_idx, bpidx)

        self.UBs[name] = ub
    def update(self, name, ub):
        PyBuffer_FromMemory = ctypes.pythonapi.PyBuffer_FromMemory
        ubo = self.UBs[name].ubo
        #print "using ubo handle %s" % ubo
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, ubo)
        vp = gl.glMapBuffer(gl.GL_UNIFORM_BUFFER, gl.GL_WRITE_ONLY)
        #buffer = PyBuffer_FromMemory(
        #    ctypes.c_void_p(vp), vbo.size
        #)

        to_p = ctypes.c_void_p(vp)
        from_p = ctypes.c_void_p(ub.listy.ctypes.data)
        #print to_p, from_p
        ctypes.memmove(to_p,
                       from_p,
                       ub.record_byte_size*len(ub.listy))
        gl.glUnmapBuffer(gl.GL_UNIFORM_BUFFER)


class ProgBundle(object):
    def __init__(self, vershade, fragshade):
        self.vershade = vershade
        self.fragshade = fragshade
        self.program = loadShaders(self.vershade, self.fragshade)
        self.tex = VTex(self.program)
        self.unif = VUnif(self.program)
        self.ublock = VUBlock(self.program)

d2r = lambda deg: (deg/180.0)*m.pi

def sph_cart(lon, lat):
    "spherical to cartesian mapping"
    lon = d2r(lon)
    lat = d2r(lat)

    # newer
    #x = abs(m.sin(lat))*m.cos(lon)
    #y = abs(m.sin(lat))*m.sin(lon)
    #z = m.cos(lat)*sgn(lat)

    # oldest
    #x = m.cos(lon)*m.cos(phi)
    #y = m.sin(theta)*m.cos(phi)
    #z = m.sin(phi)

    # newest
    rsl = m.cos(lat)
    y = m.sin(lat)
    x = rsl*m.cos(lon)
    z = rsl*m.sin(lon)

    return (x, y, z)

sgn = lambda x: x != 0 and x/abs(x) or 0
