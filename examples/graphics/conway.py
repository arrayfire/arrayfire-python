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
import array
from time import time

h_kernel = array.array('f', (1, 1, 1, 1, 0, 1, 1, 1, 1))
reset = 500
game_w = 128
game_h = 128
fps = 30

print("Example demonstrating conway's game of life using arrayfire")
print("The conway_pretty example visualizes all the states in Conway")
print("Red   : Cells that have died due to under population"    )
print("Yellow: Cells that continue to live from previous state" )
print("Green : Cells that are new as a result of reproduction"  )
print("Blue  : Cells that have died due to over population"     )
print("This examples is throttled to 30 FPS so as to be a better visualization")

simple_win = af.Window(512, 512, "Conway's Game of Life - Current State")
pretty_win = af.Window(512, 512, "Conway's Game of Life - Current State with visualization")

simple_win.set_pos(25, 15)
pretty_win.set_pos(600, 25)
frame_count = 0

# Copy kernel that specifies neighborhood conditions
kernel = af.Array(h_kernel, dims=(3,3))

# Generate the initial state with 0s and 1s
state = (af.randu(game_h, game_w) > 0.4).as_type(af.Dtype.f32)

# tile 3 times to display color
display  = af.tile(state, 1, 1, 3, 1)

while (not simple_win.close()) and (not pretty_win.close()):
    delay = time()
    if (not simple_win.close()): simple_win.image(state)
    if (not pretty_win.close()): pretty_win.image(display)

    frame_count += 1
    if (frame_count % reset == 0):
        state = (af.randu(game_h, game_w) > 0.4).as_type(af.Dtype.f32)

    neighborhood = af.convolve(state, kernel)

    # state == 1 && neighborhood <  2 --> state = 0
    # state == 1 && neighborhood >  3 --> state = 0
    # state == 0 && neighborhood == 3 --> state = 1
    # else state remains un changed

    C0 = neighborhood == 2
    C1 = neighborhood == 3
    A0 = (state == 1) & (neighborhood < 2)
    A1 = (state != 0) & (C0 | C1)
    A2 = (state == 0) & C1
    A3 = (state == 1) & (neighborhood > 3)

    display = af.join(2, A0 + A1, A1 + A2, A3).as_type(af.Dtype.f32)

    state = state * C0 + C1

    while(time() - delay < (1.0 / fps)):
        pass
