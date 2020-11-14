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


def test_simple_data() -> None:
    assert af.constant(100, 3, 3, dtype=af.Dtype.f32)
    assert af.constant(25, 3, 3, dtype=af.Dtype.c32)
    assert af.constant(2**50, 3, 3, dtype=af.Dtype.s64)
    assert af.constant(2+3j, 3, 3)
    assert af.constant(3+5j, 3, 3, dtype=af.Dtype.c32)

    assert af.range(3, 3)
    assert af.iota(3, 3, tile_dims=(2, 2))

    assert af.identity(3, 3, 1, 2, af.Dtype.b8)
    assert af.identity(3, 3, dtype=af.Dtype.c32)

    a = af.randu(3, 4)
    b = af.diag(a, extract=True)
    c = af.diag(a, 1, extract=True)

    assert a
    assert b
    assert c

    assert af.diag(b, extract=False)
    assert af.diag(c, 1, extract=False)

    assert af.join(0, a, a)
    assert af.join(1, a, a, a)

    assert af.tile(a, 2, 2)

    assert af.reorder(a, 1, 0)

    assert af.shift(a, -1, 1)

    assert af.moddims(a, 6, 2)

    assert af.flat(a)

    assert af.flip(a, 0)
    assert af.flip(a, 1)

    assert af.lower(a, False)
    assert af.lower(a, True)

    assert af.upper(a, False)
    assert af.upper(a, True)

    a = af.randu(5, 5)
    assert af.transpose(a)
    af.transpose_inplace(a)
    assert a

    assert af.select(a > 0.3, a, -0.3)

    af.replace(a, a > 0.3, -0.3)
    assert a

    assert af.pad(a, (1, 1, 0, 0), (2, 2, 0, 0))
