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

def simple_random(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    display_func(af.randu(3, 3, 1, 2))
    display_func(af.randu(3, 3, 1, 2, af.Dtype.b8))
    display_func(af.randu(3, 3, dtype=af.Dtype.c32))

    display_func(af.randn(3, 3, 1, 2))
    display_func(af.randn(3, 3, dtype=af.Dtype.c32))

    af.set_seed(1024)
    assert(af.get_seed() == 1024)

    engine = af.Random_Engine(af.RANDOM_ENGINE.MERSENNE_GP11213, 100)

    display_func(af.randu(3, 3, 1, 2, engine=engine))
    display_func(af.randu(3, 3, 1, 2, af.Dtype.s32, engine=engine))
    display_func(af.randu(3, 3, dtype=af.Dtype.c32, engine=engine))

    display_func(af.randn(3, 3, engine=engine))
    engine.set_seed(100)
    assert(engine.get_seed() == 100)

_util.tests['random'] = simple_random
