import random
from numpy import *
from numpy.linalg import *
import collada
import numpy
#import scipy
import json

import OpenGL.GL as gl
import OpenGL.GLUT as glut
from vispy.gloo import Program, VertexBuffer, IndexBuffer, Texture2D
import scrunlib

from pprint import pprint as pp

A = array

"""
>>> each = lambda x, y: [y(z) for z in x] and None or None
>>> each(i.values(), lambda x: pp(x))
"""
each = lambda x, y: [y(z) for z in x] and None or None

mag = lambda x: sqrt((x*x).sum())

ang = lambda x, y: arccos( dot(x,y)/(mag(x)*mag(y)) )

deg = lambda x: (x/pi)*180

tombstone_path = "/Users/kujawa/Desktop/3d experiments/tomb_skp.dae"
threep_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/3piece.dae"
copter_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/copter.dae"
derpcopter_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/derpcopter.dae"
simple_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/simple_center_rot.dae"
simplemat_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/simple_no_single_matrix.dae"
arrow_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/arrow.dae"
overlay_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/overlay.dae"
ballbearing_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/ballbearing.dae"
texcube_path = "/Users/kujawa/Projects/CatHouseStudio/design/model_workflow/texture_cube.dae"

# debugging stuff
def printnodes(col):
    for n in col.scene.nodes:
        if type(n) == collada.scene.Node:
            pn(n)

def pn(n, d=0):
    for c in n.children:
        if type(c) == collada.scene.Node:
            pn(c, d+1)
    print " "*d, n.id


#texstein = "/Users/kujawa/Desktop/stein/stein_texture_nonurbs.dae"
#tomb = "/Users/kujawa/Projects/CatHouseStudio/python/models/tomb_skp.dae"

# load the first geo primitive from a dae file
def loadCollada(model="models/cube.dae"):
    col = collada.Collada(model,
                          ignore=[collada.DaeUnsupportedError])
    return col

def parseCollada(col):
    # 23-jul-2013 I think this is a broken debug/dev function,
    # should be fixed or killed
    prim = col.geometries[0].primitives[0]
    shape=True
    if shape:
        numverts = len(prim.vertex)

        # normalize down to unit length
        #mag = lambda v: sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        #m = max([mag(x) for x in prim.vertex])
        #verts = prim.vertex/m

        #move it back 2 in z so it fits in our clipping plane
        #verts = verts - [0, 0, 2]
        verts = verts.flatten()
        numnorm = len(prim.normal)
        norms = prim.normal.flatten()
        cols = array([ [random.random(), 0, 0, 1] for i in
                       range(len(prim.vertex))]).flatten()
        idx = numpy.int16(prim.vertex_index).flatten()

        retval = (numverts, verts,
                  numnorm, norms,
                  numverts, cols,
                  len(idx), idx)
        #print "python returning: %s" % str(retval)
        return retval
    else:
        # debug code to be removed
        #retval = shoop()
        retval = gooder()
        print "python returning: %s" % str(retval)
        return retval

def loadGeom(path):
    return parseCollada(loadCollada(path))



# ------------------ multinode stuff -------------------------

class Geom(object):
    def digest(self):
        v, f, o = scrunlib.geomBuffer(self)
        #print v, f, o
        self.cooked_faces = v
        self.cooked_ind = f
        self.VertBuf = VertexBuffer(v)
        self.IndBuf = IndexBuffer(f)
    def twaddle_uv(self, U = False, V = False):
        u = U and 1 or 0
        v = V and 1 or 0
        if(U and V):
            tex_coord = [[u-x, v-y] for [x, y] in  self.tex_coord[0]]
        elif U:
            tex_coord = [[u-x, y] for [x, y] in  self.tex_coord[0]]
        elif V:
            tex_coord = [[x, v-y] for [x, y] in  self.tex_coord[0]]
        self.tex_coord = (numpy.array(tex_coord, dtype=numpy.float32), )
        self.digest()
    def __init__(self, verts, inds, norms, cols, tex_coord=None, tex_coord_ind=None, material=None):
        self.verts = verts
        self.inds = inds
        self.norms = norms
        self.cols = cols
        self.scale = [1.0, 1.0, 1.0]
        self.tex_coord = tex_coord
        self.tex_coord_ind = tex_coord_ind
        self.material = material
        self.reorder()
        self.digest()
    def __repr__(self):
        return "\n".join([
            "Geometry:",
            " verts: %d" % len(self.verts),
            " inds: %d" % len(self.inds),
            " norms: %d" % len(self.norms),
            " cols: %d" % len(self.cols),
            " scale: %s" % self.scale,
            " tex_coord: %s" % (self.tex_coord and str(len(self.tex_coord[0])) or "none"),
            " tex_coord_ind: %s" % (self.tex_coord_ind and str(len(self.tex_coord_ind[0])) or "none"),
            " material: %s" % self.material,
            ])
    def geom(self):
        return self.mgeom
    def marshal_geom(self):
        "translation function to make Geom object easily-digestible by objective c"

        """
        old world here
        verts = (self.verts * self.scale).flatten() # may
        numverts = len(verts)
        numnorm = len(self.norms)
        norms = (self.norms * self.scale).flatten() # need to
        cols = self.cols.flatten() # flatten
        idx = numpy.int16(self.inds).flatten() # these

        # for texture, only take the first channel for right now
        tex_coord = self.tex_coord[0].flatten()
        tex_coord_ind = self.tex_coord_ind[0].flatten()

        retval = (numverts, verts,
                  numnorm, norms,
                  numverts, cols,
                  len(idx), idx,
                  len(tex_coord), tex_coord,
                  len(tex_coord_ind), tex_coord_ind,
                  self.material.pack()
                  )
        """
        # reorder vertex array so that we can use only the one index
        revert = self.revert
        numverts = len(revert)
        numtri = len(self.tex_coord[0])
        verts = (A([v.vert() for v in revert])*self.scale).flatten()
        numnorm = len(self.norms)
        norms = (self.norms * self.scale).flatten()
        cols = A([v.col() for v in revert]).flatten()
        tex_coord_ind = self.tex_coord_ind[0].flatten()
        idx = tex_coord_ind # duplicate, as we've duplicated vertex data
        tex_coord = A([v.tc() for v in revert]).flatten()
        retval = (numverts, verts,
                  numnorm, norms,
                  numverts, cols,
                  len(idx), idx,
                  len(tex_coord), tex_coord,
                  len(tex_coord_ind), tex_coord_ind,
                  self.material.pack()
                  )
        return retval
    def reorder(self):
        self.revert = order_geom(self)
        self.mgeom = self.marshal_geom()

"""
def pf(strng, args=())
    print string % args
  def visit(self, node, printfn=pf, depth=0):
      pf(node)
      for n in node.children:
          self.visit(n, printfn, depth+1)
"""

class Node(object):
    def __init__(self, name, mat, geoms=[], children=[], transforms = {}, choad=None):
        self.name = name
        self.mat = mat
        self.geoms = geoms
        self.children = children
        self.transforms = transforms
        self.choad = choad
    def __repr__(self):
        #return "Node %s with %d geoms and %d children\n  Matrix: \n%s\n" % (self.name, len(self.geoms), len(self.children), self.mat)
        return "Node %s with %d geoms and %d children" % (self.name, len(self.geoms), len(self.children))
    def debug(self, depth=0):
        if self.name.find("__") != 0:
            print "%sdrawing: %s[%d]" % (" "*depth, self.name, len(self.geoms))
        #for g in self.geoms:
        #    print g.geom()
        for child in self.children:
            child.debug(depth+1)
    def index(self, retval={}):
        retval[self.name]=self
        for c in self.children:
            c.index(retval)
        return retval
    def handles(self):
        # returns dictionary of manipulable nodes
        ind = self.index()
        keys = [m for m in ind.keys() if m.find("m_") == 0]
        vals = [ind[k] for k in keys]
        return dict(zip(keys, vals))

# collada / materials / effect / (diffuse specular ambient etc)
# if diffuse (or whatever) is a Map:
# diffuse / sampler / (magfilter, minfilter, surface)
# surface / (format, image)
# image / getUintArray / getFloatArray
# materials indexed by name, m.name or m.id (should be same)

# each mesh has one material
# each vertex in the mesh has a coordinate in the material

# TriangleSet.texcoord_indexset
#

# g = bbl.geoms()[0]
# >>> g.material.image.pilimage.size
# (322, 482)
# >>> len(g.material.image.data)
# 224048
# >>> g.material.format
# 'A8R8G8B8'

class Texture(object):
    def __init__(self, height, width, data, image):
        self.data = data.flatten()
        # image is laid out as height columns of width rows of [rgb]
        self.size = len(self.data)
        self.width = width
        self.height = height
        self.image = image
    def __repr__(self):
        return "Texture Width: %d  Height: %d  Len: %d elements" % (self.width, self. height, self.size)
    def pack(self):
        return (self.width, self.height, self.size, self.data)

class Lambert(object):
    # 0 = None
    # 1 = float / trans
    # 2 = tuple / color
    # 3 = map / texture
    def repack_image(self, img):
        out = []
        extranum = len(img[0]) % 4
        extra = [(0,0,0,0) for x in range(extranum)]

        for y, row in enumerate(reversed(img)):
            nr = []
            for x, elt in enumerate(row):
                nr.append( (elt[0], elt[1], elt[2], 255) )
            nr = nr+extra
            out.append(nr)

        return extranum, A(out)

    def loadParm(self, parm):
        if parm.__class__ == float:
            return (1, parm)
        elif parm.__class__ == tuple:
            return (2, parm)
        elif parm.__class__ == collada.material.Map:
            surfmat = parm.sampler.surface
            fi = surfmat.image.path.find("file://")
            if(fi != -1):
                surfmat.image.path = surfmat.image.path[7:]
            #print "FIFIFIFIFIF: %s, %s" % (str(fi), str(surfmat))
            #print "%s" % str(surfmat.image)
            #arr = surfmat.image.getUintArray()
            #print "arr: %s" % str(arr)
            #raise Exception(surfmat)
            padding, data = self.repack_image(surfmat.image.getUintArray())
            pil = surfmat.image.pilimage
            (width, height) = pil.size
            return (3, Texture(height, width+padding, data, surfmat.image))
        else: # parm.type == None
            return (0, None)

    def __init__(self):
        self.emission = (0, None)
        self.ambient = (0, None)
        self.diffuse = (0, None)
        self.reflective = (0, None)
        self.transparency = (0, None)
        self.index_of_refraction = (0, None)

    def __repr__(self):
        return "\n".join([
            "\n * emission: %s" % str(self.emission),
            " * ambient: %s" % str(self.ambient),
            " * diffuse: %s" % str(self.diffuse),
            " * reflective: %s" % str(self.reflective),
            " * transparency: %s" % str(self.transparency),
            " * index_of_refraction: %s" % str(self.index_of_refraction),
            ])
    def pack(self):
        o = []
        h = {ord('e'): self.emission,
             ord('a'): self.ambient,
             ord('d'): self.diffuse,
             ord('r'): self.reflective,
             ord('t'): self.transparency,
             ord('i'): self.index_of_refraction}
        for (k, v) in h.iteritems():
            tag = v[0]
            thing = v[1]
            if (tag == 3):
                o.append( (k, tag, thing.pack() ))
            else:
                o.append( (k, tag, thing) )
        return tuple(o)

def parseNode(node, col, depth=0, par_trans={}):
    #print "%snode name: %s" % (" "*depth, node.id)
    #print dir(node)
    #print "transforms: %s" % node.transforms
    is_ident = (node.matrix == eye(4)).all()
    if is_ident:
        #print "%sidentity matrix" % (" "*depth)
        pass
    else:
        pass
        #print "%sunique matrix" % (" "*depth)
    #print "%snode matrix: %s" % (" "*depth, node.matrix)

    transf = {}
    for trans in node.transforms:
        tr = {}
        if type(trans) == collada.scene.ScaleTransform:
            tr = {'x': trans.x, 'y': trans.y, 'z': trans.z}
        elif type(trans) == collada.scene.TranslateTransform:
            tr = {'x': trans.x, 'y': trans.y, 'z': trans.z}
        elif type(trans) == collada.scene.RotateTransform:
            tr = {'x': trans.x, 'y': trans.y, 'z': trans.z, 'angle': trans.angle}
        elif type(trans) == collada.scene.MatrixTransform:
            tr = {'matrix': trans.matrix}
        elif type(trans) == collada.scene.LookAtTransform:
            tr = {'eye': trans.eye, 'interest': trans.interest, 'upvector': trans.upvector}
        else:
            raise Exception, "dunno 'bout %s" % type(trans)
        transf[trans.xmlnode.get('sid')] = tr
    if transf == {}:
        # we didn't set anything, use our parent transforms
        transf = par_trans

    #move everything back to origin wrt rotation axis
    tv = numpy.array([0,0,0])
    if transf.has_key('rotatePivot'):
        p = transf['rotatePivot'];
        tv = numpy.array([p['x'], p['y'], p['z']])

    geo = [n for n in node.children if n.__class__ == collada.scene.GeometryNode]
    geoms = []
    for g in geo:
        #print "%s> geometry: %s" % (" "*depth, g.geometry)
        for nmesh, mesh in enumerate(g.geometry.primitives):
            #print "%s> Mesh %d vertices: %d" % (" "*depth, nmesh, len(mesh.vertex))
            #print "%s> Mesh %d indices: %d" % (" "*depth, nmesh, len(mesh.vertex_index))
            #print "%s> Mesh %d normals: %d" % (" "*depth, nmesh, len(mesh.normal))
            fect = col.materials[mesh.material].effect
            lambert = Lambert()

            diffuse = fect.diffuse
            #print "diffuse: %s [%s]" % (diffuse, diffuse.__class__)
            lambert.diffuse = lambert.loadParm(diffuse)

            emission = fect.emission
            #print "emission: %s [%s]" % (emission, emission.__class__)
            lambert.emission = lambert.loadParm(emission)

            ambient = fect.ambient
            #print "ambient: %s [%s]" % (ambient, ambient.__class__)
            lambert.ambient = lambert.loadParm(ambient)

            transparency = fect.transparency
            #print "transparency: %s [%s]" % (transparency, transparency.__class__)
            lambert.transparency = lambert.loadParm(transparency)

            #reflective = fect.reflective
            #print "reflective: %s [%s]" % (reflective, reflective.__class__)

            #index_of_refraction = fect.index_of_refraction
            #print "index_of_refraction: %s [%s]" % (index_of_refraction, index_of_refraction.__class__)



            #print "%s> Mesh %d diffuse: %s" % (" "*depth, nmesh, str(diffuse))


            # try an make everything diffuse, see if this barfs
            if type(diffuse) == type(()): # we actually have a color
                #print "tuple diffuse: %s [%s]" % (diffuse, diffuse.__class__)
                d = numpy.array(diffuse, dtype=numpy.float32)
                cols = numpy.array([d for v in mesh.vertex], dtype=numpy.float32)
            else:
                #make everything grey by default if it doesn't fit into the new world order
                cols = array([(0.5, 0.5, 0.5, 1.0) for v in mesh.vertex])

            surfmat = None
            tci = None
            tc = None
            try:
                tci = mesh.texcoord_indexset
                tc = mesh.texcoordset
            except Exception, e:
                print "texture not loaded: %s" % e
            geoms.append(Geom(mesh.vertex-tv, mesh.vertex_index, mesh.normal, cols, tc, tci, lambert))
    children = [parseNode(n, col, depth+1, transf) for n in node.children if n.__class__ == node.__class__]
    return Node(node.id, node.matrix, geoms, children, transf, node)
    #return Node(node.id, eye(4), geoms, children)
def parseScene(col):
    for snum, scene in enumerate(col.scenes):
        print "scene %d" % snum
        for node in scene.nodes:
            return parseNode(node, col)

def loadNodeScene(path):
    return parseScene(loadCollada(path))

def bbl(path=texcube_path):
    n = loadNodeScene(path)
    n.draw()
    #mz = n.handles()['m_z']
    #print "m_z transforms"
    #pp(mz.transforms)
    print "nodes"
    pp(sorted(n.index().keys()))
    #g = n.index()['group']
    #print "group transforms"
    #pp(g.transforms)
    #rds = n.index()['RealDWG_Shape_Rep']
    #print "rds transforms"
    #pp(rds.transforms)
    print
    for node in sorted(n.index().keys()):
        if(n.index()[node].transforms):
            print node
            pp(n.index()[node].transforms)
    # matrix json serialization
    js = json.dumps({"mat": list(numpy.eye(4).flatten())})
    mat = array(json.loads(js)['mat']).reshape(4,4)
    print mat
    return n

class Vertex(object):
    def __init__(self, x=0, y=0, z=0, r=0, g=0, b=0, a=0, u=0, v=0):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.a = a
        self.u = u
        self.v = v
    def __repr__(self):
        s = "".join(["XYZ: (%0.2f %0.2f %0.2f) " % (self.x, self.y, self.z),
                     "RGBA: (%0.2f %0.2f %0.2f %0.2f) " % (self.r, self.g, self.b, self.a),
                     "UV: (%0.5f %0.5f) " % (self.u, self.v)])
        return s
    def vert(self):
        return ([self.x, self.y, self.z])
    def col(self):
        return ([self.r, self.g, self.b, self.a])
    def tc(self):
        return ([self.u, self.v])


def order_geom(g):
    tci = g.tex_coord_ind[0]
    tc = g.tex_coord[0]
    #print "output array will be %d members" % len(tc)
    output = [Vertex() for x in range(len(tc))]
    #print "index, tex_coord_ind, vert_ind"
    for i, tci_i in enumerate(tci):
        vind = g.inds[i]
        #print i, tci_i, vind
        for j, tc_ind in enumerate(tci_i):
            vert_ind = vind[j]
            #print j, tc_ind, vert_ind
            output[tc_ind].x = g.verts[vert_ind][0]
            output[tc_ind].y = g.verts[vert_ind][1]
            output[tc_ind].z = g.verts[vert_ind][2]

            output[tc_ind].r = g.cols[vert_ind][0]
            output[tc_ind].g = g.cols[vert_ind][1]
            output[tc_ind].b = g.cols[vert_ind][2]
            output[tc_ind].a = g.cols[vert_ind][3]

            output[tc_ind].u = tc[tc_ind][0]
            output[tc_ind].v = tc[tc_ind][1]
    #print "output array:"
    #for i, v in enumerate(output):
    #    print i, v
    return output

"""
        print "verts: ",
        for vi in vind:
            print g.verts[vi],
        print
        print "texture coords: ",
        for ti in tci:
            print g.tex_coord[0][ti],
        print
        print
        """

def penum(o):
    for i, j in enumerate(o):
        print "%s, %s" % (i, ", ".join([str(x) for x in j]))

def img2csv(img):
    ua = img.getUintArray()
    head = ', '.join([str(x) for x in range(len(ua))])
    f = open("/Users/kujawa/Projects/CatHouseStudio/repolink/testbed/texbox.csv", "w")
    print >>f, "r, "+head
    for y, row in enumerate(ua):
        print >>f, "%d, " % y,
        for x, elt in enumerate(row):
            print >>f,  "(%0.2f %0.2f %0.2f), " % (elt[0], elt[1], elt[2]),
        print >>f, ""
