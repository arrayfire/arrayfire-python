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

    a = af.randu(10, 1)
    pos0 = af.randu(10) * 10
    display_func(af.approx1(a, pos0))

    a = af.randu(3, 3)
    pos0 = af.randu(3, 3) * 10
    pos1 = af.randu(3, 3) * 10

    display_func(af.approx2(a, pos0, pos1))

    a = af.randu(8, 1)
    display_func(a)

    display_func(af.fft(a))
    display_func(af.dft(a))
    display_func(af.real(af.ifft(af.fft(a))))
    display_func(af.real(af.idft(af.dft(a))))

    a = af.randu(4, 4)
    display_func(a)

    display_func(af.fft2(a))
    display_func(af.dft(a))
    display_func(af.real(af.ifft2(af.fft2(a))))
    display_func(af.real(af.idft(af.dft(a))))

    a = af.randu(4, 4, 2)
    display_func(a)

    display_func(af.fft3(a))
    display_func(af.dft(a))
    display_func(af.real(af.ifft3(af.fft3(a))))
    display_func(af.real(af.idft(af.dft(a))))

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

_util.tests['signal'] = simple_signal
