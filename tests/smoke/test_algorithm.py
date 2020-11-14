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


def test_algorithm() -> None:
    a = af.randu(3, 3)
    k = af.constant(1, 3, 3, dtype=af.Dtype.u32)
    af.eval(k)

    assert af.sum(a)
    assert af.product(a)
    assert af.min(a)
    assert af.max(a)
    assert af.count(a)
    assert af.any_true(a)
    assert af.all_true(a)

    assert af.sum(a, 0)
    assert af.sum(a, 1)

    rk = af.constant(1, 3, dtype=af.Dtype.u32)
    rk[2] = 0
    af.eval(rk)
    assert af.sumByKey(rk, a, dim=0)
    assert af.sumByKey(rk, a, dim=1)

    assert af.productByKey(rk, a, dim=0)
    assert af.productByKey(rk, a, dim=1)

    assert af.minByKey(rk, a, dim=0)
    assert af.minByKey(rk, a, dim=1)

    assert af.maxByKey(rk, a, dim=0)
    assert af.maxByKey(rk, a, dim=1)

    assert af.anyTrueByKey(rk, a, dim=0)
    assert af.anyTrueByKey(rk, a, dim=1)

    assert af.allTrueByKey(rk, a, dim=0)
    assert af.allTrueByKey(rk, a, dim=1)

    assert af.countByKey(rk, a, dim=0)
    assert af.countByKey(rk, a, dim=1)

    assert af.product(a, 0)
    assert af.product(a, 1)

    assert af.min(a, 0)
    assert af.min(a, 1)

    assert af.max(a, 0)
    assert af.max(a, 1)

    assert af.count(a, 0)
    assert af.count(a, 1)

    assert af.any_true(a, 0)
    assert af.any_true(a, 1)

    assert af.all_true(a, 0)
    assert af.all_true(a, 1)

    assert af.accum(a, 0)
    assert af.accum(a, 1)

    assert af.scan(a, 0, af.BINARYOP.ADD)
    assert af.scan(a, 1, af.BINARYOP.MAX)

    assert af.scan_by_key(k, a, 0, af.BINARYOP.ADD)
    assert af.scan_by_key(k, a, 1, af.BINARYOP.MAX)

    assert af.sort(a, is_ascending=True)
    assert af.sort(a, is_ascending=False)

    b = (a > 0.1) * a
    c = (a > 0.4) * a
    d = b / c
    assert af.sum(d)
    assert af.sum(d, nan_val=0.0)
    assert af.sum(d, dim=0, nan_val=0.0)

    val, idx = af.sort_index(a, is_ascending=True)
    assert val
    assert idx
    val, idx = af.sort_index(a, is_ascending=False)
    assert val
    assert idx

    b = af.randu(3, 3)
    keys, vals = af.sort_by_key(a, b, is_ascending=True)
    assert keys
    assert vals
    keys, vals = af.sort_by_key(a, b, is_ascending=False)
    assert keys
    assert vals

    c = af.randu(5, 1)
    d = af.randu(5, 1)
    cc = af.set_unique(c, is_sorted=False)
    dd = af.set_unique(af.sort(d), is_sorted=True)
    assert cc
    assert dd

    assert af.set_union(cc, dd, is_unique=True)
    assert af.set_union(cc, dd, is_unique=False)

    assert af.set_intersect(cc, cc, is_unique=True)
    assert af.set_intersect(cc, cc, is_unique=False)
