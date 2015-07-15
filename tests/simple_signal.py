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

a = af.randu(10, 1)
pos0 = af.randu(10) * 10
af.print_array(af.approx1(a, pos0))

a = af.randu(3, 3)
pos0 = af.randu(3, 3) * 10
pos1 = af.randu(3, 3) * 10

af.print_array(af.approx2(a, pos0, pos1))

a = af.randu(8, 1)
af.print_array(a)

af.print_array(af.fft(a))
af.print_array(af.dft(a))
af.print_array(af.real(af.ifft(af.fft(a))))
af.print_array(af.real(af.idft(af.dft(a))))

a = af.randu(4, 4)
af.print_array(a)

af.print_array(af.fft2(a))
af.print_array(af.dft(a))
af.print_array(af.real(af.ifft2(af.fft2(a))))
af.print_array(af.real(af.idft(af.dft(a))))

a = af.randu(4, 4, 2)
af.print_array(a)

af.print_array(af.fft3(a))
af.print_array(af.dft(a))
af.print_array(af.real(af.ifft3(af.fft3(a))))
af.print_array(af.real(af.idft(af.dft(a))))

a = af.randu(10, 1)
b = af.randu(3, 1)
af.print_array(af.convolve1(a, b))
af.print_array(af.fft_convolve1(a, b))
af.print_array(af.convolve(a, b))
af.print_array(af.fft_convolve(a, b))

a = af.randu(5, 5)
b = af.randu(3, 3)
af.print_array(af.convolve2(a, b))
af.print_array(af.fft_convolve2(a, b))
af.print_array(af.convolve(a, b))
af.print_array(af.fft_convolve(a, b))

a = af.randu(5, 5, 3)
b = af.randu(3, 3, 2)
af.print_array(af.convolve3(a, b))
af.print_array(af.fft_convolve3(a, b))
af.print_array(af.convolve(a, b))
af.print_array(af.fft_convolve(a, b))


b = af.randu(3, 1)
x = af.randu(10, 1)
a = af.randu(2, 1)
af.print_array(af.fir(b, x))
af.print_array(af.iir(b, a, x))
