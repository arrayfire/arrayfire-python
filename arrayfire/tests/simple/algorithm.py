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

def simple_algorithm(verbose = False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)

    a = af.randu(3, 3)
    k = af.constant(1, 3, 3, dtype=af.Dtype.u32)
    af.eval(k)

    print_func(af.sum(a), af.product(a), af.min(a), af.max(a),
               af.count(a), af.any_true(a), af.all_true(a))

    display_func(af.sum(a, 0))
    display_func(af.sum(a, 1))

    display_func(af.product(a, 0))
    display_func(af.product(a, 1))

    display_func(af.min(a, 0))
    display_func(af.min(a, 1))

    display_func(af.max(a, 0))
    display_func(af.max(a, 1))

    display_func(af.count(a, 0))
    display_func(af.count(a, 1))

    display_func(af.any_true(a, 0))
    display_func(af.any_true(a, 1))

    display_func(af.all_true(a, 0))
    display_func(af.all_true(a, 1))

    display_func(af.accum(a, 0))
    display_func(af.accum(a, 1))

    display_func(af.scan(a, 0, af.BINARYOP.ADD))
    display_func(af.scan(a, 1, af.BINARYOP.MAX))

    display_func(af.scan_by_key(k, a, 0, af.BINARYOP.ADD))
    display_func(af.scan_by_key(k, a, 1, af.BINARYOP.MAX))

    display_func(af.sort(a, is_ascending=True))
    display_func(af.sort(a, is_ascending=False))

    b = (a > 0.1) * a
    c = (a > 0.4) * a
    d = b / c
    print_func(af.sum(d));
    print_func(af.sum(d, nan_val=0.0));
    display_func(af.sum(d, dim=0, nan_val=0.0));

    val,idx = af.sort_index(a, is_ascending=True)
    display_func(val)
    display_func(idx)
    val,idx = af.sort_index(a, is_ascending=False)
    display_func(val)
    display_func(idx)

    b = af.randu(3,3)
    keys,vals = af.sort_by_key(a, b, is_ascending=True)
    display_func(keys)
    display_func(vals)
    keys,vals = af.sort_by_key(a, b, is_ascending=False)
    display_func(keys)
    display_func(vals)

    c = af.randu(5,1)
    d = af.randu(5,1)
    cc = af.set_unique(c, is_sorted=False)
    dd = af.set_unique(af.sort(d), is_sorted=True)
    display_func(cc)
    display_func(dd)

    display_func(af.set_union(cc, dd, is_unique=True))
    display_func(af.set_union(cc, dd, is_unique=False))

    display_func(af.set_intersect(cc, cc, is_unique=True))
    display_func(af.set_intersect(cc, cc, is_unique=False))

_util.tests['algorithm'] = simple_algorithm
