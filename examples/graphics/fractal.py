#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af
import sys
from math import sqrt

width = 400
height = 400

def complex_grid(w, h, zoom, center):
    x = (af.iota(d0 = 1, d1 = h, tile_dims = (w, 1)) - h/2) / zoom + center[0]
    y = (af.iota(d0 = w, d1 = 1, tile_dims = (1, h)) - w/2) / zoom + center[1]
    return af.cplx(x, y)

def mandelbrot(data, it, maxval):
    C = data
    Z = data
    mag = af.constant(0, *C.dims())

    for ii in range(1, 1 + it):
        # Doing the calculation
        Z = Z * Z + C

        # Get indices where abs(Z) crosses maxval
        cond = ((af.abs(Z) > maxval)).as_type(af.Dtype.f32)
        mag = af.maxof(mag, cond * ii)

        C = C * (1 - cond)
        Z = Z * (1 - cond)

        af.eval(C)
        af.eval(Z)

    return mag / maxval

def normalize(a):
    mx = af.max(a)
    mn = af.min(a)
    return (a - mn)/(mx - mn)

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))

    af.info()

    print("ArrayFire Fractal Demo\n")

    win = af.Window(width, height, "Fractal Demo")
    win.set_colormap(af.COLORMAP.SPECTRUM)

    center = (-0.75, 0.1)

    for i in range(10, 400):
        zoom = i * i
        if not (i % 10):
            print("Iteration: %d zoom: %d" % (i, zoom))

        c = complex_grid(width, height, zoom, center)
        it = sqrt(2*sqrt(abs(1-sqrt(5*zoom))))*100

        if (win.close()): break
        mag = mandelbrot(c, int(it), 1000)

        win.image(normalize(mag))
