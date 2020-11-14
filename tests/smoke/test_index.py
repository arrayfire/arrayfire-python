#!/usr/bin/env python

#######################################################
# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import array as host

import arrayfire as af


def test_simple_index() -> None:
    a = af.randu(5, 5)
    assert a
    b = af.Array(a)
    assert b

    c = a.copy()
    assert c
    assert a[0, 0]
    assert a[0]
    assert a[:]
    assert a[:, :]
    assert a[0:3, ]
    assert a[-2:-1, -1]
    assert a[0:5]
    assert a[0:5:2]

    idx = af.Array(host.array("i", [0, 3, 2]))
    assert idx
    aa = a[idx]
    assert aa

    a[0] = 1
    assert a
    a[0] = af.randu(1, 5)
    assert a
    a[:] = af.randu(5, 5)
    assert a
    a[:, -1] = af.randu(5, 1)
    assert a
    a[0:5:2] = af.randu(3, 5)
    assert a
    a[idx, idx] = af.randu(3, 3)
    assert a

    a = af.randu(5, 1)
    b = af.randu(5, 1)
    assert a
    assert b
    for ii in af.ParallelRange(1, 3):
        a[ii] = b[ii]

    assert a

    for ii in af.ParallelRange(2, 5):
        b[ii] = 2
    assert b

    a = af.randu(3, 2)
    rows = af.constant(0, 1, dtype=af.Dtype.s32)
    b = a[:, rows]
    assert b
    for r in range(rows.elements()):
        assert b[:, r]

    a = af.randu(3)
    c = af.randu(3)
    b = af.constant(1, 3, dtype=af.Dtype.b8)
    assert a
    a[b] = c
    assert a
