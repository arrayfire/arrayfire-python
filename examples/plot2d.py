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
import math

POINTS = 10000
PRECISION = 1.0 / float(POINTS)

val = -math.pi
X = math.pi * (2 * (af.range(POINTS) / POINTS) - 1)

win = af.Window(512, 512, "2D Plot example using ArrayFire")
sign = 1.0

while not win.close():
    Y = af.sin(X)
    win.plot(X, Y)

    X += PRECISION * sign
    val += PRECISION * sign

    if (val > math.pi):
        sign = -1.0
    elif (val < -math.pi):
        sign = 1.0
