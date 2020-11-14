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


def test_simple_signal() -> None:
    signal = af.randu(10)
    x_new = af.randu(10)
    x_orig = af.randu(10)
    assert af.approx1(signal, x_new, xp=x_orig)

    signal = af.randu(3, 3)
    x_new = af.randu(3, 3)
    x_orig = af.randu(3, 3)
    y_new = af.randu(3, 3)
    y_orig = af.randu(3, 3)

    assert af.approx2(signal, x_new, y_new, xp=x_orig, yp=y_orig)

    a = af.randu(8, 1)
    assert a

    assert af.fft(a)
    assert af.dft(a)
    assert af.real(af.ifft(af.fft(a)))
    assert af.real(af.idft(af.dft(a)))

    b = af.fft(a)
    af.ifft_inplace(b)
    assert b
    af.fft_inplace(b)
    assert b

    b = af.fft_r2c(a)
    c = af.fft_c2r(b)
    assert b
    assert c

    a = af.randu(4, 4)
    assert a

    assert af.fft2(a)
    assert af.dft(a)
    assert af.real(af.ifft2(af.fft2(a)))
    assert af.real(af.idft(af.dft(a)))

    b = af.fft2(a)
    af.ifft2_inplace(b)
    assert b
    af.fft2_inplace(b)
    assert b

    b = af.fft2_r2c(a)
    c = af.fft2_c2r(b)
    assert b
    assert c

    a = af.randu(4, 4, 2)
    assert a

    assert af.fft3(a)
    assert af.dft(a)
    assert af.real(af.ifft3(af.fft3(a)))
    assert af.real(af.idft(af.dft(a)))

    b = af.fft3(a)
    af.ifft3_inplace(b)
    assert b
    af.fft3_inplace(b)
    assert b

    b = af.fft3_r2c(a)
    c = af.fft3_c2r(b)
    assert b
    assert c

    a = af.randu(10, 1)
    b = af.randu(3, 1)
    assert af.convolve1(a, b)
    assert af.fft_convolve1(a, b)
    assert af.convolve(a, b)
    assert af.fft_convolve(a, b)

    a = af.randu(5, 5)
    b = af.randu(3, 3)
    assert af.convolve2(a, b)
    assert af.fft_convolve2(a, b)
    assert af.convolve(a, b)
    assert af.fft_convolve(a, b)

    c = af.convolve2NN(a, b)
    assert c
    in_dims = c.dims()
    incoming_grad = af.constant(1, in_dims[0], in_dims[1]);
    g = af.convolve2GradientNN(incoming_grad, a, b, c)
    assert g

    a = af.randu(5, 5, 3)
    b = af.randu(3, 3, 2)
    assert af.convolve3(a, b)
    assert af.fft_convolve3(a, b)
    assert af.convolve(a, b)
    assert af.fft_convolve(a, b)

    b = af.randu(3, 1)
    x = af.randu(10, 1)
    a = af.randu(2, 1)
    assert af.fir(b, x)
    assert af.iir(b, a, x)

    assert af.medfilt1(a)
    assert af.medfilt2(a)
    assert af.medfilt(a)
