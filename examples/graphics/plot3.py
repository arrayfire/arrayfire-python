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

ITERATIONS = 200
POINTS = int(10.0 * ITERATIONS)

Z = 1 + af.range(POINTS) / ITERATIONS

win = af.Window(800, 800, "3D Plot example using ArrayFire")

t = 0.1
while not win.close():
    X = af.cos(Z * t + t) / Z
    Y = af.sin(Z * t + t) / Z

    X = af.maxof(af.minof(X, 1), -1)
    Y = af.maxof(af.minof(Y, 1), -1)

    Pts = af.join(1, X, Y, Z)
    win.plot3(Pts)
    t = t + 0.01
