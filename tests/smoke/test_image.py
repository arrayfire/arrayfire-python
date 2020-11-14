#!/usr/bin/env python

#######################################################
# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af


def test_simple_image() -> None:
    a = 10 * af.randu(6, 6)
    a3 = 10 * af.randu(5, 5, 3)

    dx, dy = af.gradient(a)
    assert dx
    assert dy

    assert af.resize(a, scale=0.5)
    assert af.resize(a, odim0=8, odim1=8)

    t = af.randu(3, 2)
    assert af.transform(a, t)
    assert af.rotate(a, 3.14)
    assert af.translate(a, 1, 1)
    assert af.scale(a, 1.2, 1.2, 7, 7)
    assert af.skew(a, 0.02, 0.02)
    h = af.histogram(a, 3)
    assert h
    assert af.hist_equal(a, h)

    assert af.dilate(a)
    assert af.erode(a)

    assert af.dilate3(a3)
    assert af.erode3(a3)

    assert af.bilateral(a, 1, 2)
    assert af.mean_shift(a, 1, 2, 3)

    assert af.medfilt(a)
    assert af.minfilt(a)
    assert af.maxfilt(a)

    assert af.regions(af.round(a) > 3)
    assert af.confidenceCC(
        af.randu(10, 10), (af.randu(2) * 9).as_type(af.Dtype.u32), (af.randu(2) * 9).as_type(af.Dtype.u32),
        3, 3, 10, 0.1)

    dx, dy = af.sobel_derivatives(a)
    assert dx
    assert dy
    assert af.sobel_filter(a)
    assert af.gaussian_kernel(3, 3)
    assert af.gaussian_kernel(3, 3, 1, 1)

    ac = af.gray2rgb(a)
    assert ac
    assert af.rgb2gray(ac)
    ah = af.rgb2hsv(ac)
    assert ah
    assert af.hsv2rgb(ah)

    assert af.color_space(a, af.CSPACE.RGB, af.CSPACE.GRAY)

    a = af.randu(6, 6)
    b = af.unwrap(a, 2, 2, 2, 2)
    c = af.wrap(b, 6, 6, 2, 2, 2, 2)
    assert a
    assert b
    assert c
    assert af.sat(a)

    a = af.randu(10, 10, 3)
    assert af.rgb2ycbcr(a)
    assert af.ycbcr2rgb(a)

    a = af.randu(10, 10)
    b = af.canny(a, low_threshold=0.2, high_threshold=0.8)

    # FIXME: OpenCL Error (-11): Build Program Failure when calling clBuildProgram
    # assert af.anisotropic_diffusion(a, 0.125, 1.0, 64, af.FLUX.QUADRATIC, af.DIFFUSION.GRAD)

    a = af.randu(10, 10)
    psf = af.gaussian_kernel(3, 3)
    cimg = af.convolve(a, psf)
    assert af.iterativeDeconv(cimg, psf, 100, 0.5, af.ITERATIVE_DECONV.LANDWEBER)
    assert af.iterativeDeconv(cimg, psf, 100, 0.5, af.ITERATIVE_DECONV.RICHARDSONLUCY)
    assert af.inverseDeconv(cimg, psf, 1.0, af.INVERSE_DECONV.TIKHONOV)
