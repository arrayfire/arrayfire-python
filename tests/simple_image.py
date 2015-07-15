#!/usr/bin/python
#######################################################
# Copyright (c) 2014, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af

a = 10 * af.randu(6, 6)
a3 = 10 * af.randu(5,5,3)

dx,dy = af.gradient(a)
af.print_array(dx)
af.print_array(dy)

af.print_array(af.resize(a, scale=0.5))
af.print_array(af.resize(a, odim0=8, odim1=8))

t = af.randu(3,2)
af.print_array(af.transform(a, t))
af.print_array(af.rotate(a, 3.14))
af.print_array(af.translate(a, 1, 1))
af.print_array(af.scale(a, 1.2, 1.2, 7, 7))
af.print_array(af.skew(a, 0.02, 0.02))
h = af.histogram(a, 3)
af.print_array(h)
af.print_array(af.hist_equal(a, h))

af.print_array(af.dilate(a))
af.print_array(af.erode(a))

af.print_array(af.dilate3(a3))
af.print_array(af.erode3(a3))

af.print_array(af.bilateral(a, 1, 2))
af.print_array(af.mean_shift(a, 1, 2, 3))

af.print_array(af.medfilt(a))
af.print_array(af.minfilt(a))
af.print_array(af.maxfilt(a))

af.print_array(af.regions(af.round(a) > 3))

dx,dy = af.sobel_derivatives(a)
af.print_array(dx)
af.print_array(dy)
af.print_array(af.sobel_filter(a))

ac = af.gray2rgb(a)
af.print_array(ac)
af.print_array(af.rgb2gray(ac))
ah = af.rgb2hsv(ac)
af.print_array(ah)
af.print_array(af.hsv2rgb(ah))

af.print_array(af.color_space(a, af.AF_RGB, af.AF_GRAY))
