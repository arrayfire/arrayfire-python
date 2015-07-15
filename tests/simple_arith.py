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

a = af.randu(3,3,dtype=af.u32)
b = af.constant(4, 3, 3, dtype=af.u32)
af.print_array(a)
af.print_array(b)

c = a + b
d = a
d += b

af.print_array(c)
af.print_array(d)
af.print_array(a + 2)
af.print_array(3 + a)


c = a - b
d = a
d -= b

af.print_array(c)
af.print_array(d)
af.print_array(a - 2)
af.print_array(3 - a)

c = a * b
d = a
d *= b

af.print_array(c * 2)
af.print_array(3 * d)
af.print_array(a * 2)
af.print_array(3 * a)

c = a / b
d = a
d /= b

af.print_array(c / 2.0)
af.print_array(3.0 / d)
af.print_array(a / 2)
af.print_array(3 / a)

c = a % b
d = a
d %= b

af.print_array(c % 2.0)
af.print_array(3.0 % d)
af.print_array(a % 2)
af.print_array(3 % a)

c = a ** b
d = a
d **= b

af.print_array(c ** 2.0)
af.print_array(3.0 ** d)
af.print_array(a ** 2)
af.print_array(3 ** a)

af.print_array(a < b)
af.print_array(a < 0.5)
af.print_array(0.5 < a)

af.print_array(a <= b)
af.print_array(a <= 0.5)
af.print_array(0.5 <= a)

af.print_array(a > b)
af.print_array(a > 0.5)
af.print_array(0.5 > a)

af.print_array(a >= b)
af.print_array(a >= 0.5)
af.print_array(0.5 >= a)

af.print_array(a != b)
af.print_array(a != 0.5)
af.print_array(0.5 != a)

af.print_array(a == b)
af.print_array(a == 0.5)
af.print_array(0.5 == a)

af.print_array(a & b)
af.print_array(a & 2)
c = a
c &= 2
af.print_array(c)

af.print_array(a | b)
af.print_array(a | 2)
c = a
c |= 2
af.print_array(c)

af.print_array(a >> b)
af.print_array(a >> 2)
c = a
c >>= 2
af.print_array(c)

af.print_array(a << b)
af.print_array(a << 2)
c = a
c <<= 2
af.print_array(c)

af.print_array(-a)
af.print_array(+a)
af.print_array(~a)
af.print_array(a)

af.print_array(af.cast(a, af.c32))
af.print_array(af.maxof(a,b))
af.print_array(af.minof(a,b))
af.print_array(af.rem(a,b))

a = af.randu(3,3) - 0.5
b = af.randu(3,3) - 0.5

af.print_array(af.abs(a))
af.print_array(af.arg(a))
af.print_array(af.sign(a))
af.print_array(af.round(a))
af.print_array(af.trunc(a))
af.print_array(af.floor(a))
af.print_array(af.ceil(a))
af.print_array(af.hypot(a, b))
af.print_array(af.sin(a))
af.print_array(af.cos(a))
af.print_array(af.tan(a))
af.print_array(af.asin(a))
af.print_array(af.acos(a))
af.print_array(af.atan(a))
af.print_array(af.atan2(a, b))

c = af.cplx(a)
d = af.cplx(a,b)
af.print_array(c)
af.print_array(d)
af.print_array(af.real(d))
af.print_array(af.imag(d))
af.print_array(af.conjg(d))

af.print_array(af.sinh(a))
af.print_array(af.cosh(a))
af.print_array(af.tanh(a))
af.print_array(af.asinh(a))
af.print_array(af.acosh(a))
af.print_array(af.atanh(a))

a = af.abs(a)
b = af.abs(b)

af.print_array(af.root(a, b))
af.print_array(af.pow(a, b))
af.print_array(af.pow2(a))
af.print_array(af.exp(a))
af.print_array(af.expm1(a))
af.print_array(af.erf(a))
af.print_array(af.erfc(a))
af.print_array(af.log(a))
af.print_array(af.log1p(a))
af.print_array(af.log10(a))
af.print_array(af.log2(a))
af.print_array(af.sqrt(a))
af.print_array(af.cbrt(a))

a = af.round(5 * af.randu(3,3) - 1)
b = af.round(5 * af.randu(3,3) - 1)

af.print_array(af.factorial(a))
af.print_array(af.tgamma(a))
af.print_array(af.lgamma(a))
af.print_array(af.iszero(a))
af.print_array(af.isinf(a/b))
af.print_array(af.isnan(a/a))
