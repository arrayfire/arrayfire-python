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


def test_simple_random() -> None:
    assert af.randu(3, 3, 1, 2)
    assert af.randu(3, 3, 1, 2, af.Dtype.b8)
    assert af.randu(3, 3, dtype=af.Dtype.c32)

    assert af.randn(3, 3, 1, 2)
    assert af.randn(3, 3, dtype=af.Dtype.c32)

    af.set_seed(1024)
    assert af.get_seed() == 1024

    engine = af.Random_Engine(af.RANDOM_ENGINE.MERSENNE_GP11213, 100)

    assert af.randu(3, 3, 1, 2, engine=engine)
    assert af.randu(3, 3, 1, 2, af.Dtype.s32, engine=engine)
    assert af.randu(3, 3, dtype=af.Dtype.c32, engine=engine)

    assert af.randn(3, 3, engine=engine)
    engine.set_seed(100)
    assert engine.get_seed() == 100
