#!/usr/bin/python

import arrayfire as af
import math

POINTS = 10000
PRECISION = 1.0 / float(POINTS)

val = -math.pi
X = math.pi * (2 * (af.range(POINTS) / POINTS) - 1)

win = af.window(512, 512, "2D Plot example using ArrayFire")
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
