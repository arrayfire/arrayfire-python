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
af.display(af.approx1(a, pos0))

a = af.randu(3, 3)
pos0 = af.randu(3, 3) * 10
pos1 = af.randu(3, 3) * 10

af.display(af.approx2(a, pos0, pos1))

a = af.randu(8, 1)
af.display(a)

af.display(af.fft(a))
af.display(af.dft(a))
af.display(af.real(af.ifft(af.fft(a))))
af.display(af.real(af.idft(af.dft(a))))

a = af.randu(4, 4)
af.display(a)

af.display(af.fft2(a))
af.display(af.dft(a))
af.display(af.real(af.ifft2(af.fft2(a))))
af.display(af.real(af.idft(af.dft(a))))

a = af.randu(4, 4, 2)
af.display(a)

af.display(af.fft3(a))
af.display(af.dft(a))
af.display(af.real(af.ifft3(af.fft3(a))))
af.display(af.real(af.idft(af.dft(a))))

a = af.randu(10, 1)
b = af.randu(3, 1)
af.display(af.convolve1(a, b))
af.display(af.fft_convolve1(a, b))
af.display(af.convolve(a, b))
af.display(af.fft_convolve(a, b))

a = af.randu(5, 5)
b = af.randu(3, 3)
af.display(af.convolve2(a, b))
af.display(af.fft_convolve2(a, b))
af.display(af.convolve(a, b))
af.display(af.fft_convolve(a, b))

a = af.randu(5, 5, 3)
b = af.randu(3, 3, 2)
af.display(af.convolve3(a, b))
af.display(af.fft_convolve3(a, b))
af.display(af.convolve(a, b))
af.display(af.fft_convolve(a, b))


b = af.randu(3, 1)
x = af.randu(10, 1)
a = af.randu(2, 1)
af.display(af.fir(b, x))
af.display(af.iir(b, a, x))
