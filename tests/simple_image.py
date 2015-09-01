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

a = 10 * af.randu(6, 6)
a3 = 10 * af.randu(5,5,3)

dx,dy = af.gradient(a)
af.display(dx)
af.display(dy)

af.display(af.resize(a, scale=0.5))
af.display(af.resize(a, odim0=8, odim1=8))

t = af.randu(3,2)
af.display(af.transform(a, t))
af.display(af.rotate(a, 3.14))
af.display(af.translate(a, 1, 1))
af.display(af.scale(a, 1.2, 1.2, 7, 7))
af.display(af.skew(a, 0.02, 0.02))
h = af.histogram(a, 3)
af.display(h)
af.display(af.hist_equal(a, h))

af.display(af.dilate(a))
af.display(af.erode(a))

af.display(af.dilate3(a3))
af.display(af.erode3(a3))

af.display(af.bilateral(a, 1, 2))
af.display(af.mean_shift(a, 1, 2, 3))

af.display(af.medfilt(a))
af.display(af.minfilt(a))
af.display(af.maxfilt(a))

af.display(af.regions(af.round(a) > 3))

dx,dy = af.sobel_derivatives(a)
af.display(dx)
af.display(dy)
af.display(af.sobel_filter(a))

ac = af.gray2rgb(a)
af.display(ac)
af.display(af.rgb2gray(ac))
ah = af.rgb2hsv(ac)
af.display(ah)
af.display(af.hsv2rgb(ah))

af.display(af.color_space(a, af.CSPACE.RGB, af.CSPACE.GRAY))
