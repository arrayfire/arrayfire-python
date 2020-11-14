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


def test_simple_arith() -> None:
    a = af.randu(3, 3)
    b = af.constant(4, 3, 3)
    assert a
    assert b

    c = a + b
    d = a
    d += b

    assert c
    assert d
    assert a + 2
    assert 3 + a

    c = a - b
    d = a
    d -= b

    assert c
    assert d
    assert a - 2
    assert 3 - a

    c = a * b
    d = a
    d *= b

    assert c * 2
    assert 3 * d
    assert a * 2
    assert 3 * a

    c = a / b
    d = a
    d /= b

    assert c / 2.0
    assert 3.0 / d
    assert a / 2
    assert 3 / a

    c = a % b
    d = a
    d %= b

    assert c % 2.0
    assert 3.0 % d
    assert a % 2
    assert 3 % a

    c = a ** b
    d = a
    d **= b

    assert c ** 2.0
    assert 3.0 ** d
    assert a ** 2
    assert 3 ** a

    assert a < b
    assert a < 0.5
    assert a > 0.5

    assert a <= b
    assert a <= 0.5
    assert a >= 0.5

    assert a > b
    assert a > 0.5
    assert a < 0.5

    assert a >= b
    assert a >= 0.5
    assert a <= 0.5

    assert a != b
    assert a != 0.5

    assert a == b
    assert a == 0.5

    a = af.randu(3, 3, dtype=af.Dtype.u32)
    b = af.constant(4, 3, 3, dtype=af.Dtype.u32)

    assert a & b
    assert a & 2
    c = a
    c &= 2
    assert c

    assert a | b
    assert a | 2
    c = a
    c |= 2
    assert c

    assert a >> b
    assert a >> 2
    c = a
    c >>= 2
    assert c

    assert a << b
    assert a << 2
    c = a
    c <<= 2
    assert c

    assert -a
    assert +a
    assert ~a
    assert a

    assert af.cast(a, af.Dtype.c32)
    assert af.maxof(a, b)
    assert af.minof(a, b)

    assert af.clamp(a, 0, 1)
    assert af.clamp(a, 0, b)
    assert af.clamp(a, b, 1)

    assert af.rem(a, b)

    a = af.randu(3, 3) - 0.5
    b = af.randu(3, 3) - 0.5

    assert af.abs(a)
    assert af.arg(a)
    assert af.sign(a)
    assert af.round(a)
    assert af.trunc(a)
    assert af.floor(a)
    assert af.ceil(a)
    assert af.hypot(a, b)
    assert af.sin(a)
    assert af.cos(a)
    assert af.tan(a)
    assert af.asin(a)
    assert af.acos(a)
    assert af.atan(a)
    assert af.atan2(a, b)

    c = af.cplx(a)
    d = af.cplx(a, b)
    assert c
    assert d
    assert af.real(d)
    assert af.imag(d)
    assert af.conjg(d)

    assert af.sinh(a)
    assert af.cosh(a)
    assert af.tanh(a)
    assert af.asinh(a)
    assert af.acosh(a)
    assert af.atanh(a)

    a = af.abs(a)
    b = af.abs(b)

    assert af.root(a, b)
    assert af.pow(a, b)
    assert af.pow2(a)
    assert af.sigmoid(a)
    assert af.exp(a)
    assert af.expm1(a)
    assert af.erf(a)
    assert af.erfc(a)
    assert af.log(a)
    assert af.log1p(a)
    assert af.log10(a)
    assert af.log2(a)
    assert af.sqrt(a)
    assert af.rsqrt(a)
    assert af.cbrt(a)

    a = af.round(5 * af.randu(3, 3) - 1)
    b = af.round(5 * af.randu(3, 3) - 1)

    assert af.factorial(a)
    assert af.tgamma(a)
    assert af.lgamma(a)
    assert af.iszero(a)
    assert af.isinf(a/b)
    assert af.isnan(a/a)

    a = af.randu(5, 1)
    b = af.randu(1, 5)
    c = af.broadcast(lambda x, y: x+y, a, b)
    assert a
    assert b
    assert c

    @af.broadcast
    def test_add(aa, bb):
        return aa + bb

    assert test_add(a, b)
