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


def test_simple_blas() -> None:
    a = af.randu(5, 5)
    b = af.randu(5, 5)

    assert af.matmul(a, b)
    assert af.matmul(a, b, af.MATPROP.TRANS)
    assert af.matmul(a, b, af.MATPROP.NONE, af.MATPROP.TRANS)

    b = af.randu(5, 1)
    assert af.dot(b, b)
