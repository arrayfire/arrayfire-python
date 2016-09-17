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

def simple_blas(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)
    a = af.randu(5,5)
    b = af.randu(5,5)

    display_func(af.matmul(a,b))
    display_func(af.matmul(a,b,af.MATPROP.TRANS))
    display_func(af.matmul(a,b,af.MATPROP.NONE, af.MATPROP.TRANS))

    b = af.randu(5,1)
    display_func(af.dot(b,b))

_util.tests['blas'] = simple_blas
