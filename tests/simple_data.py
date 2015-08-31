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

af.display(af.constant(100, 3,3, dtype=af.f32))
af.display(af.constant(25, 3,3, dtype=af.c32))
af.display(af.constant(2**50, 3,3, dtype=af.s64))
af.display(af.constant(2+3j, 3,3))
af.display(af.constant(3+5j, 3,3, dtype=af.c32))

af.display(af.range(3, 3))
af.display(af.iota(3, 3, tile_dims=(2,2)))

af.display(af.randu(3, 3, 1, 2))
af.display(af.randu(3, 3, 1, 2, af.b8))
af.display(af.randu(3, 3, dtype=af.c32))

af.display(af.randn(3, 3, 1, 2))
af.display(af.randn(3, 3, dtype=af.c32))

af.set_seed(1024)
assert(af.get_seed() == 1024)

af.display(af.identity(3, 3, 1, 2, af.b8))
af.display(af.identity(3, 3, dtype=af.c32))

a = af.randu(3, 4)
b = af.diag(a, extract=True)
c = af.diag(a, 1, extract=True)

af.display(a)
af.display(b)
af.display(c)

af.display(af.diag(b, extract = False))
af.display(af.diag(c, 1, extract = False))

af.display(af.join(0, a, a))
af.display(af.join(1, a, a, a))

af.display(af.tile(a, 2, 2))


af.display(af.reorder(a, 1, 0))

af.display(af.shift(a, -1, 1))

af.display(af.moddims(a, 6, 2))

af.display(af.flat(a))

af.display(af.flip(a, 0))
af.display(af.flip(a, 1))

af.display(af.lower(a, False))
af.display(af.lower(a, True))

af.display(af.upper(a, False))
af.display(af.upper(a, True))
