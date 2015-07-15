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

af.print_array(af.constant(100, 3,3, dtype=af.f32))
af.print_array(af.constant(25, 3,3, dtype=af.c32))
af.print_array(af.constant(2**50, 3,3, dtype=af.s64))
af.print_array(af.constant(2+3j, 3,3))
af.print_array(af.constant(3+5j, 3,3, dtype=af.c32))

af.print_array(af.range(3, 3))
af.print_array(af.iota(3, 3, tile_dims=(2,2)))

af.print_array(af.randu(3, 3, 1, 2))
af.print_array(af.randu(3, 3, 1, 2, af.b8))
af.print_array(af.randu(3, 3, dtype=af.c32))

af.print_array(af.randn(3, 3, 1, 2))
af.print_array(af.randn(3, 3, dtype=af.c32))

af.set_seed(1024)
assert(af.get_seed() == 1024)

af.print_array(af.identity(3, 3, 1, 2, af.b8))
af.print_array(af.identity(3, 3, dtype=af.c32))

a = af.randu(3, 4)
b = af.diag(a, extract=True)
c = af.diag(a, 1, extract=True)

af.print_array(a)
af.print_array(b)
af.print_array(c)

af.print_array(af.diag(b, extract = False))
af.print_array(af.diag(c, 1, extract = False))

af.print_array(af.tile(a, 2, 2))

af.print_array(af.reorder(a, 1, 0))

af.print_array(af.shift(a, -1, 1))

af.print_array(af.moddims(a, 6, 2))

af.print_array(af.flat(a))

af.print_array(af.flip(a, 0))
af.print_array(af.flip(a, 1))

af.print_array(af.lower(a, False))
af.print_array(af.lower(a, True))

af.print_array(af.upper(a, False))
af.print_array(af.upper(a, True))
