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
from array import array

def af_assert(left, right, eps=1E-6):
    if (af.max(af.abs(left -right)) > eps):
        raise ValueError("Arrays not within dictated precision")
    return

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    af.info()

    h_dx = array('f', (1.0/12, -8.0/12, 0, 8.0/12, 1.0/12))
    h_spread = array('f', (1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5))

    img = af.randu(640, 480)
    dx = af.Array(h_dx, (5,1))
    spread = af.Array(h_spread, (1, 5))
    kernel = af.matmul(dx, spread)

    full_res = af.convolve2(img, kernel)
    sep_res = af.convolve2_separable(dx, spread, img)

    af_assert(full_res, sep_res)

    print("full      2D convolution time: %.5f ms" %
          (1000 * af.timeit(af.convolve2, img, kernel)))
    print("separable 2D convolution time: %.5f ms" %
          (1000 * af.timeit(af.convolve2_separable, dx, spread, img)))
