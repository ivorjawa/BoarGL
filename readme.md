# Boar Modern-ish Python OpenGL/OpenCL Wrapper

An attempt to demonstrate various OpenGL 3.2+ techniques, providing a clean python library simplifying many of OpenGL's functions.

## List of components

`transforms.py` comes from [Vispy](http://vispy.org)
`transformations.py` and its accompanying .so library come from [Christoph Golke](http://www.lfd.uci.edu/~gohlke/code/transformations.py.html)
Code was adapted from [Adventures in PyOpenCL: Part 2, Particles with PyOpenGL](http://enja.org/2011/03/22/adventures-in-pyopencl-part-2-particles-with-pyopengl/)

## Concerns

1. The `ub_binding_count` code in `BoarGL/glbase.py` is sketchy as hell, but it seems to work
