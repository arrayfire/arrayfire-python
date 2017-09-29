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

def simple_signal(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    signal = af.randu(10)
    x_new  = af.randu(10)
    x_orig = af.randu(10)
    display_func(af.approx1(signal, x_new, xp = x_orig))

    signal = af.randu(3, 3)
    x_new  = af.randu(3, 3)
    x_orig = af.randu(3, 3)
    y_new  = af.randu(3, 3)
    y_orig = af.randu(3, 3)

    display_func(af.approx2(signal, x_new, y_new, xp = x_orig, yp = y_orig))

    a = af.randu(8, 1)
    display_func(a)

    display_func(af.fft(a))
    display_func(af.dft(a))
    display_func(af.real(af.ifft(af.fft(a))))
    display_func(af.real(af.idft(af.dft(a))))

    b = af.fft(a)
    af.ifft_inplace(b)
    display_func(b)
    af.fft_inplace(b)
    display_func(b)

    b = af.fft_r2c(a)
    c = af.fft_c2r(b)
    display_func(b)
    display_func(c)

    a = af.randu(4, 4)
    display_func(a)

    display_func(af.fft2(a))
    display_func(af.dft(a))
    display_func(af.real(af.ifft2(af.fft2(a))))
    display_func(af.real(af.idft(af.dft(a))))

    b = af.fft2(a)
    af.ifft2_inplace(b)
    display_func(b)
    af.fft2_inplace(b)
    display_func(b)

    b = af.fft2_r2c(a)
    c = af.fft2_c2r(b)
    display_func(b)
    display_func(c)

    a = af.randu(4, 4, 2)
    display_func(a)

    display_func(af.fft3(a))
    display_func(af.dft(a))
    display_func(af.real(af.ifft3(af.fft3(a))))
    display_func(af.real(af.idft(af.dft(a))))

    b = af.fft3(a)
    af.ifft3_inplace(b)
    display_func(b)
    af.fft3_inplace(b)
    display_func(b)

    b = af.fft3_r2c(a)
    c = af.fft3_c2r(b)
    display_func(b)
    display_func(c)

    a = af.randu(10, 1)
    b = af.randu(3, 1)
    display_func(af.convolve1(a, b))
    display_func(af.fft_convolve1(a, b))
    display_func(af.convolve(a, b))
    display_func(af.fft_convolve(a, b))

    a = af.randu(5, 5)
    b = af.randu(3, 3)
    display_func(af.convolve2(a, b))
    display_func(af.fft_convolve2(a, b))
    display_func(af.convolve(a, b))
    display_func(af.fft_convolve(a, b))

    a = af.randu(5, 5, 3)
    b = af.randu(3, 3, 2)
    display_func(af.convolve3(a, b))
    display_func(af.fft_convolve3(a, b))
    display_func(af.convolve(a, b))
    display_func(af.fft_convolve(a, b))


    b = af.randu(3, 1)
    x = af.randu(10, 1)
    a = af.randu(2, 1)
    display_func(af.fir(b, x))
    display_func(af.iir(b, a, x))

    display_func(af.medfilt1(a))
    display_func(af.medfilt2(a))
    display_func(af.medfilt(a))

_util.tests['signal'] = simple_signal
