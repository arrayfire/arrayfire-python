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


def test_simple_array() -> None:
    a = af.array.Array([1, 2, 3])
    assert a
    assert a.T
    assert a.H
    assert a.shape

    b = a.as_type(af.Dtype.s32)
    assert b

    assert b.elements()
    assert b.type()
    assert b.dims()
    assert b.numdims()
    assert not b.is_empty()
    assert not b.is_scalar()
    assert b.is_column()
    assert not b.is_row()
    assert not b.is_complex()
    assert b.is_real()
    assert not b.is_double()
    assert not b.is_single()
    assert not b.is_real_floating()
    assert not b.is_floating()
    assert b.is_integer()
    assert not b.is_bool()

    a = af.array.Array(host.array("i", [4, 5, 6]))
    assert a
    assert a.elements()
    assert a.type()
    assert a.dims()
    assert a.numdims()
    assert not a.is_empty()
    assert not a.is_scalar()
    assert a.is_column()
    assert not a.is_row()
    assert not a.is_complex()
    assert a.is_real()
    assert not a.is_double()
    assert not a.is_single()
    assert not a.is_real_floating()
    assert not a.is_floating()
    assert a.is_integer()
    assert not a.is_bool()

    a = af.array.Array(host.array("I", [7, 8, 9] * 3), (3, 3))
    assert a
    assert a.elements()
    assert a.type()
    assert a.dims()
    assert a.numdims()
    assert not a.is_empty()
    assert not a.is_scalar()
    assert not a.is_column()
    assert not a.is_row()
    assert not a.is_complex()
    assert a.is_real()
    assert not a.is_double()
    assert not a.is_single()
    assert not a.is_real_floating()
    assert not a.is_floating()
    assert a.is_integer()
    assert not a.is_bool()

    c = a.to_ctype()
    for n in range(a.elements()):
        assert c[n]

    c, s = a.to_ctype(True, True)
    for n in range(a.elements()):
        assert c[n]
    assert s

    arr = a.to_array()
    lst = a.to_list(True)

    assert arr
    assert lst

    assert not a.is_sparse()
