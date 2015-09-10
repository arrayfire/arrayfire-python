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

def simple_statistics(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    a = af.randu(5, 5)
    b = af.randu(5, 5)
    w = af.randu(5, 1)

    display_func(af.mean(a, dim=0))
    display_func(af.mean(a, weights=w, dim=0))
    print_func(af.mean(a))
    print_func(af.mean(a, weights=w))

    display_func(af.var(a, dim=0))
    display_func(af.var(a, isbiased=True, dim=0))
    display_func(af.var(a, weights=w, dim=0))
    print_func(af.var(a))
    print_func(af.var(a, isbiased=True))
    print_func(af.var(a, weights=w))

    display_func(af.stdev(a, dim=0))
    print_func(af.stdev(a))

    display_func(af.var(a, dim=0))
    display_func(af.var(a, isbiased=True, dim=0))
    print_func(af.var(a))
    print_func(af.var(a, isbiased=True))

    display_func(af.median(a, dim=0))
    print_func(af.median(w))

    print_func(af.corrcoef(a, b))

_util.tests['statistics'] = simple_statistics
