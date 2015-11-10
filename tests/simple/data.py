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

def simple_data(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    display_func(af.constant(100, 3,3, dtype=af.Dtype.f32))
    display_func(af.constant(25, 3,3, dtype=af.Dtype.c32))
    display_func(af.constant(2**50, 3,3, dtype=af.Dtype.s64))
    display_func(af.constant(2+3j, 3,3))
    display_func(af.constant(3+5j, 3,3, dtype=af.Dtype.c32))

    display_func(af.range(3, 3))
    display_func(af.iota(3, 3, tile_dims=(2,2)))

    display_func(af.randu(3, 3, 1, 2))
    display_func(af.randu(3, 3, 1, 2, af.Dtype.b8))
    display_func(af.randu(3, 3, dtype=af.Dtype.c32))

    display_func(af.randn(3, 3, 1, 2))
    display_func(af.randn(3, 3, dtype=af.Dtype.c32))

    af.set_seed(1024)
    assert(af.get_seed() == 1024)

    display_func(af.identity(3, 3, 1, 2, af.Dtype.b8))
    display_func(af.identity(3, 3, dtype=af.Dtype.c32))

    a = af.randu(3, 4)
    b = af.diag(a, extract=True)
    c = af.diag(a, 1, extract=True)

    display_func(a)
    display_func(b)
    display_func(c)

    display_func(af.diag(b, extract = False))
    display_func(af.diag(c, 1, extract = False))

    display_func(af.join(0, a, a))
    display_func(af.join(1, a, a, a))

    display_func(af.tile(a, 2, 2))


    display_func(af.reorder(a, 1, 0))

    display_func(af.shift(a, -1, 1))

    display_func(af.moddims(a, 6, 2))

    display_func(af.flat(a))

    display_func(af.flip(a, 0))
    display_func(af.flip(a, 1))

    display_func(af.lower(a, False))
    display_func(af.lower(a, True))

    display_func(af.upper(a, False))
    display_func(af.upper(a, True))

    a = af.randu(5,5)
    display_func(af.transpose(a))
    af.transpose_inplace(a)
    display_func(a)

    display_func(af.select(a > 0.3, a, -0.3))

    af.replace(a, a > 0.3, -0.3)
    display_func(a)

_util.tests['data'] = simple_data
