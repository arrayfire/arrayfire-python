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

a = af.randu(3,3,dtype=af.u32)
b = af.constant(4, 3, 3, dtype=af.u32)
af.display(a)
af.display(b)

c = a + b
d = a
d += b

af.display(c)
af.display(d)
af.display(a + 2)
af.display(3 + a)


c = a - b
d = a
d -= b

af.display(c)
af.display(d)
af.display(a - 2)
af.display(3 - a)

c = a * b
d = a
d *= b

af.display(c * 2)
af.display(3 * d)
af.display(a * 2)
af.display(3 * a)

c = a / b
d = a
d /= b

af.display(c / 2.0)
af.display(3.0 / d)
af.display(a / 2)
af.display(3 / a)

c = a % b
d = a
d %= b

af.display(c % 2.0)
af.display(3.0 % d)
af.display(a % 2)
af.display(3 % a)

c = a ** b
d = a
d **= b

af.display(c ** 2.0)
af.display(3.0 ** d)
af.display(a ** 2)
af.display(3 ** a)

af.display(a < b)
af.display(a < 0.5)
af.display(0.5 < a)

af.display(a <= b)
af.display(a <= 0.5)
af.display(0.5 <= a)

af.display(a > b)
af.display(a > 0.5)
af.display(0.5 > a)

af.display(a >= b)
af.display(a >= 0.5)
af.display(0.5 >= a)

af.display(a != b)
af.display(a != 0.5)
af.display(0.5 != a)

af.display(a == b)
af.display(a == 0.5)
af.display(0.5 == a)

af.display(a & b)
af.display(a & 2)
c = a
c &= 2
af.display(c)

af.display(a | b)
af.display(a | 2)
c = a
c |= 2
af.display(c)

af.display(a >> b)
af.display(a >> 2)
c = a
c >>= 2
af.display(c)

af.display(a << b)
af.display(a << 2)
c = a
c <<= 2
af.display(c)

af.display(-a)
af.display(+a)
af.display(~a)
af.display(a)

af.display(af.cast(a, af.c32))
af.display(af.maxof(a,b))
af.display(af.minof(a,b))
af.display(af.rem(a,b))

a = af.randu(3,3) - 0.5
b = af.randu(3,3) - 0.5

af.display(af.abs(a))
af.display(af.arg(a))
af.display(af.sign(a))
af.display(af.round(a))
af.display(af.trunc(a))
af.display(af.floor(a))
af.display(af.ceil(a))
af.display(af.hypot(a, b))
af.display(af.sin(a))
af.display(af.cos(a))
af.display(af.tan(a))
af.display(af.asin(a))
af.display(af.acos(a))
af.display(af.atan(a))
af.display(af.atan2(a, b))

c = af.cplx(a)
d = af.cplx(a,b)
af.display(c)
af.display(d)
af.display(af.real(d))
af.display(af.imag(d))
af.display(af.conjg(d))

af.display(af.sinh(a))
af.display(af.cosh(a))
af.display(af.tanh(a))
af.display(af.asinh(a))
af.display(af.acosh(a))
af.display(af.atanh(a))

a = af.abs(a)
b = af.abs(b)

af.display(af.root(a, b))
af.display(af.pow(a, b))
af.display(af.pow2(a))
af.display(af.exp(a))
af.display(af.expm1(a))
af.display(af.erf(a))
af.display(af.erfc(a))
af.display(af.log(a))
af.display(af.log1p(a))
af.display(af.log10(a))
af.display(af.log2(a))
af.display(af.sqrt(a))
af.display(af.cbrt(a))

a = af.round(5 * af.randu(3,3) - 1)
b = af.round(5 * af.randu(3,3) - 1)

af.display(af.factorial(a))
af.display(af.tgamma(a))
af.display(af.lgamma(a))
af.display(af.iszero(a))
af.display(af.isinf(a/b))
af.display(af.isnan(a/a))
