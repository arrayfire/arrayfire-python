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

a = af.randu(5,5)

l,u,p = af.lu(a)

af.display(l)
af.display(u)
af.display(p)

p = af.lu_inplace(a, "full")

af.display(a)
af.display(p)

a = af.randu(5,3)

q,r,t = af.qr(a)

af.display(q)
af.display(r)
af.display(t)

af.qr_inplace(a)

af.display(a)

a = af.randu(5, 5)
a = af.matmulTN(a, a) + 10 * af.identity(5,5)

R,info = af.cholesky(a)
af.display(R)
print(info)

af.cholesky_inplace(a)
af.display(a)

a = af.randu(5,5)
ai = af.inverse(a)

af.display(a)
af.display(ai)

x0 = af.randu(5, 3)
b = af.matmul(a, x0)
x1 = af.solve(a, b)

af.display(x0)
af.display(x1)

p = af.lu_inplace(a)

x2 = af.solve_lu(a, p, b)

af.display(x2)

print(af.rank(a))
print(af.det(a))
print(af.norm(a, af.AF_NORM_EUCLID))
print(af.norm(a, af.AF_NORM_MATRIX_1))
print(af.norm(a, af.AF_NORM_MATRIX_INF))
print(af.norm(a, af.AF_NORM_MATRIX_L_PQ, 1, 1))
