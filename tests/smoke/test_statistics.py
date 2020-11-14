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


def test_simple_statistics() -> None:
    a = af.randu(5, 5)
    b = af.randu(5, 5)
    w = af.randu(5, 1)

    assert af.mean(a, dim=0)
    assert af.mean(a, weights=w, dim=0)
    assert af.mean(a)
    assert af.mean(a, weights=w)

    assert af.var(a, dim=0)
    assert af.var(a, isbiased=True, dim=0)
    assert af.var(a, weights=w, dim=0)
    assert af.var(a)
    assert af.var(a, isbiased=True)
    assert af.var(a, weights=w)

    mean, var = af.meanvar(a, dim=0)
    assert mean
    assert var
    mean, var = af.meanvar(a, weights=w, bias=af.VARIANCE.SAMPLE, dim=0)
    assert mean
    assert var

    assert af.stdev(a, dim=0)
    assert af.stdev(a)

    assert af.var(a, dim=0)
    assert af.var(a, isbiased=True, dim=0)
    assert af.var(a)
    assert af.var(a, isbiased=True)

    assert af.median(a, dim=0)
    assert af.median(w)

    assert af.corrcoef(a, b)

    data = af.iota(5, 3)
    k = 3
    dim = 0
    order = af.TOPK.DEFAULT  # defaults to af.TOPK.MAX
    assert dim == 0  # topk currently supports first dim only
    values, indices = af.topk(data, k, dim, order)
    assert values
    assert indices
