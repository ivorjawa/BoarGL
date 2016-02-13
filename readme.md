# Boar Modern-ish Python OpenGL/OpenCL Wrapper

An attempt to demonstrate various OpenGL 3.2+ techniques, providing a clean python library simplifying many of OpenGL's functions.

Written in python 2.7.

Usage:  python spectree.py [-w] [-spectrum] [-xmas]

 * -w runs in window mode.
 * -spectrum pulses a spectrum
 * -xmas makes a 4-color christmas tree display 
 * by default listens to the default microphone and bases the light show on an FFT

[Potato-quality video of it running in FFT mode](https://www.youtube.com/watch?v=OJ_7f4Uw-wU)

## List of components

1. `transforms.py` comes from [Vispy](http://vispy.org)
2. `transformations.py` and its accompanying .so library come from [Christoph Golke](http://www.lfd.uci.edu/~gohlke/code/transformations.py.html)
3. Code was adapted from [Adventures in PyOpenCL: Part 2, Particles with PyOpenGL](http://enja.org/2011/03/22/adventures-in-pyopencl-part-2-particles-with-pyopengl/)

## Concerns

1. The `ub_binding_count` code in `BoarGL/glbase.py` is sketchy as hell, but it seems to work
