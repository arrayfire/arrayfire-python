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

def simple_lapack(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)
    a = af.randu(5,5)

    l,u,p = af.lu(a)

    display_func(l)
    display_func(u)
    display_func(p)

    p = af.lu_inplace(a, "full")

    display_func(a)
    display_func(p)

    a = af.randu(5,3)

    q,r,t = af.qr(a)

    display_func(q)
    display_func(r)
    display_func(t)

    af.qr_inplace(a)

    display_func(a)

    a = af.randu(5, 5)
    a = af.matmulTN(a, a) + 10 * af.identity(5,5)

    R,info = af.cholesky(a)
    display_func(R)
    print_func(info)

    af.cholesky_inplace(a)
    display_func(a)

    a = af.randu(5,5)
    ai = af.inverse(a)

    display_func(a)
    display_func(ai)

    x0 = af.randu(5, 3)
    b = af.matmul(a, x0)
    x1 = af.solve(a, b)

    display_func(x0)
    display_func(x1)

    p = af.lu_inplace(a)

    x2 = af.solve_lu(a, p, b)

    display_func(x2)

    print_func(af.rank(a))
    print_func(af.det(a))
    print_func(af.norm(a, af.NORM.EUCLID))
    print_func(af.norm(a, af.NORM.MATRIX_1))
    print_func(af.norm(a, af.NORM.MATRIX_INF))
    print_func(af.norm(a, af.NORM.MATRIX_L_PQ, 1, 1))

    a = af.randu(10,10)
    display_func(a)
    u,s,vt = af.svd(a)
    display_func(af.matmul(af.matmul(u, af.diag(s, 0, False)), vt))
    u,s,vt = af.svd_inplace(a)
    display_func(af.matmul(af.matmul(u, af.diag(s, 0, False)), vt))

_util.tests['lapack'] = simple_lapack
