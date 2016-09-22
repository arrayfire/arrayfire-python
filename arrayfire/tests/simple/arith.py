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

def simple_arith(verbose = False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    a = af.randu(3,3)
    b = af.constant(4, 3, 3)
    display_func(a)
    display_func(b)

    c = a + b
    d = a
    d += b

    display_func(c)
    display_func(d)
    display_func(a + 2)
    display_func(3 + a)


    c = a - b
    d = a
    d -= b

    display_func(c)
    display_func(d)
    display_func(a - 2)
    display_func(3 - a)

    c = a * b
    d = a
    d *= b

    display_func(c * 2)
    display_func(3 * d)
    display_func(a * 2)
    display_func(3 * a)

    c = a / b
    d = a
    d /= b

    display_func(c / 2.0)
    display_func(3.0 / d)
    display_func(a / 2)
    display_func(3 / a)

    c = a % b
    d = a
    d %= b

    display_func(c % 2.0)
    display_func(3.0 % d)
    display_func(a % 2)
    display_func(3 % a)

    c = a ** b
    d = a
    d **= b

    display_func(c ** 2.0)
    display_func(3.0 ** d)
    display_func(a ** 2)
    display_func(3 ** a)

    display_func(a < b)
    display_func(a < 0.5)
    display_func(0.5 < a)

    display_func(a <= b)
    display_func(a <= 0.5)
    display_func(0.5 <= a)

    display_func(a > b)
    display_func(a > 0.5)
    display_func(0.5 > a)

    display_func(a >= b)
    display_func(a >= 0.5)
    display_func(0.5 >= a)

    display_func(a != b)
    display_func(a != 0.5)
    display_func(0.5 != a)

    display_func(a == b)
    display_func(a == 0.5)
    display_func(0.5 == a)

    a = af.randu(3,3,dtype=af.Dtype.u32)
    b = af.constant(4, 3, 3, dtype=af.Dtype.u32)

    display_func(a & b)
    display_func(a & 2)
    c = a
    c &= 2
    display_func(c)

    display_func(a | b)
    display_func(a | 2)
    c = a
    c |= 2
    display_func(c)

    display_func(a >> b)
    display_func(a >> 2)
    c = a
    c >>= 2
    display_func(c)

    display_func(a << b)
    display_func(a << 2)
    c = a
    c <<= 2
    display_func(c)

    display_func(-a)
    display_func(+a)
    display_func(~a)
    display_func(a)

    display_func(af.cast(a, af.Dtype.c32))
    display_func(af.maxof(a,b))
    display_func(af.minof(a,b))

    display_func(af.clamp(a, 0, 1))
    display_func(af.clamp(a, 0, b))
    display_func(af.clamp(a, b, 1))

    display_func(af.rem(a,b))

    a = af.randu(3,3) - 0.5
    b = af.randu(3,3) - 0.5

    display_func(af.abs(a))
    display_func(af.arg(a))
    display_func(af.sign(a))
    display_func(af.round(a))
    display_func(af.trunc(a))
    display_func(af.floor(a))
    display_func(af.ceil(a))
    display_func(af.hypot(a, b))
    display_func(af.sin(a))
    display_func(af.cos(a))
    display_func(af.tan(a))
    display_func(af.asin(a))
    display_func(af.acos(a))
    display_func(af.atan(a))
    display_func(af.atan2(a, b))

    c = af.cplx(a)
    d = af.cplx(a,b)
    display_func(c)
    display_func(d)
    display_func(af.real(d))
    display_func(af.imag(d))
    display_func(af.conjg(d))

    display_func(af.sinh(a))
    display_func(af.cosh(a))
    display_func(af.tanh(a))
    display_func(af.asinh(a))
    display_func(af.acosh(a))
    display_func(af.atanh(a))

    a = af.abs(a)
    b = af.abs(b)

    display_func(af.root(a, b))
    display_func(af.pow(a, b))
    display_func(af.pow2(a))
    display_func(af.sigmoid(a))
    display_func(af.exp(a))
    display_func(af.expm1(a))
    display_func(af.erf(a))
    display_func(af.erfc(a))
    display_func(af.log(a))
    display_func(af.log1p(a))
    display_func(af.log10(a))
    display_func(af.log2(a))
    display_func(af.sqrt(a))
    display_func(af.cbrt(a))

    a = af.round(5 * af.randu(3,3) - 1)
    b = af.round(5 * af.randu(3,3) - 1)

    display_func(af.factorial(a))
    display_func(af.tgamma(a))
    display_func(af.lgamma(a))
    display_func(af.iszero(a))
    display_func(af.isinf(a/b))
    display_func(af.isnan(a/a))

    a = af.randu(5, 1)
    b = af.randu(1, 5)
    c = af.broadcast(lambda x,y: x+y, a, b)
    display_func(a)
    display_func(b)
    display_func(c)

    @af.broadcast
    def test_add(aa, bb):
        return aa + bb

    display_func(test_add(a, b))

_util.tests['arith'] = simple_arith
