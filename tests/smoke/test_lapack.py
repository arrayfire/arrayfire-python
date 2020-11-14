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


def test_simple_lapack() -> None:
    a = af.randu(5, 5)

    l, u, p = af.lu(a)

    assert l
    assert u
    assert p

    p = af.lu_inplace(a, "full")

    assert a
    assert p

    a = af.randu(5, 3)

    q, r, t = af.qr(a)

    assert q
    assert r
    assert t

    af.qr_inplace(a)

    assert a

    a = af.randu(5, 5)
    a = af.matmulTN(a, a.copy()) + 10 * af.identity(5, 5)

    R, info = af.cholesky(a)
    assert R

    af.cholesky_inplace(a)
    assert a

    a = af.randu(5, 5)
    ai = af.inverse(a)

    assert a
    assert ai

    ai = af.pinverse(a)
    assert ai

    x0 = af.randu(5, 3)
    b = af.matmul(a, x0)
    x1 = af.solve(a, b)

    assert x0
    assert x1

    p = af.lu_inplace(a)

    x2 = af.solve_lu(a, p, b)

    assert x2

    assert af.rank(a)
    assert af.det(a)
    assert af.norm(a, af.NORM.EUCLID)
    assert af.norm(a, af.NORM.MATRIX_1)
    assert af.norm(a, af.NORM.MATRIX_INF)
    assert af.norm(a, af.NORM.MATRIX_L_PQ, 1, 1)

    a = af.randu(10, 10)
    assert a
    u, s, vt = af.svd(a)
    assert af.matmul(af.matmul(u, af.diag(s, 0, False)), vt)
    u, s, vt = af.svd_inplace(a)
    assert af.matmul(af.matmul(u, af.diag(s, 0, False)), vt)
