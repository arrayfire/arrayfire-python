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
from . import _util

def simple_image(verbose = False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    a = 10 * af.randu(6, 6)
    a3 = 10 * af.randu(5,5,3)

    dx,dy = af.gradient(a)
    display_func(dx)
    display_func(dy)

    display_func(af.resize(a, scale=0.5))
    display_func(af.resize(a, odim0=8, odim1=8))

    t = af.randu(3,2)
    display_func(af.transform(a, t))
    display_func(af.rotate(a, 3.14))
    display_func(af.translate(a, 1, 1))
    display_func(af.scale(a, 1.2, 1.2, 7, 7))
    display_func(af.skew(a, 0.02, 0.02))
    h = af.histogram(a, 3)
    display_func(h)
    display_func(af.hist_equal(a, h))

    display_func(af.dilate(a))
    display_func(af.erode(a))

    display_func(af.dilate3(a3))
    display_func(af.erode3(a3))

    display_func(af.bilateral(a, 1, 2))
    display_func(af.mean_shift(a, 1, 2, 3))

    display_func(af.medfilt(a))
    display_func(af.minfilt(a))
    display_func(af.maxfilt(a))

    display_func(af.regions(af.round(a) > 3))

    dx,dy = af.sobel_derivatives(a)
    display_func(dx)
    display_func(dy)
    display_func(af.sobel_filter(a))
    display_func(af.gaussian_kernel(3, 3))
    display_func(af.gaussian_kernel(3, 3, 1, 1))

    ac = af.gray2rgb(a)
    display_func(ac)
    display_func(af.rgb2gray(ac))
    ah = af.rgb2hsv(ac)
    display_func(ah)
    display_func(af.hsv2rgb(ah))

    display_func(af.color_space(a, af.CSPACE.RGB, af.CSPACE.GRAY))

    a = af.randu(6,6)
    b = af.unwrap(a, 2, 2, 2, 2)
    c = af.wrap(b, 6, 6, 2, 2, 2, 2)
    display_func(a)
    display_func(b)
    display_func(c)
    display_func(af.sat(a))

    a = af.randu(10,10,3)
    display_func(af.rgb2ycbcr(a))
    display_func(af.ycbcr2rgb(a))

    a = af.randu(10, 10)
    b = af.canny(a, low_threshold = 0.2, high_threshold = 0.8)

    display_func(af.anisotropic_diffusion(a, 0.125, 1.0, 64, af.FLUX.QUADRATIC, af.DIFFUSION.GRAD))

_util.tests['image'] = simple_image
