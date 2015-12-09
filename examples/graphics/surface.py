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

af.info()

POINTS = 30
N = 2 * POINTS

x = (af.iota(d0 = N, d1 = 1, tile_dims = (1, N)) - POINTS) / POINTS
y = (af.iota(d0 = 1, d1 = N, tile_dims = (N, 1)) - POINTS) / POINTS

win = af.Window(800, 800, "3D Surface example using ArrayFire")

t = 0
while not win.close():
    t = t + 0.07
    z = 10*x*-af.abs(y) * af.cos(x*x*(y+t))+af.sin(y*(x+t))-1.5;
    win.surface(x, y, z)
