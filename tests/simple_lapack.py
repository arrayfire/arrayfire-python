#!/usr/bin/python
#######################################################
# Copyright (c) 2014, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
import arrayfire as af

a = af.randu(5,5)

l,u,p = af.lu(a)

af.print_array(l)
af.print_array(u)
af.print_array(p)

p = af.lu_inplace(a, "full")

af.print_array(a)
af.print_array(p)

a = af.randu(5,3)

q,r,t = af.qr(a)

af.print_array(q)
af.print_array(r)
af.print_array(t)

af.qr_inplace(a)

af.print_array(a)

a = af.randu(5, 5)
a = af.matmulTN(a, a) + 10 * af.identity(5,5)

R,info = af.cholesky(a)
af.print_array(R)
print(info)

af.cholesky_inplace(a)
af.print_array(a)

a = af.randu(5,5)
ai = af.inverse(a)

af.print_array(a)
af.print_array(ai)

x0 = af.randu(5, 3)
b = af.matmul(a, x0)
x1 = af.solve(a, b)

af.print_array(x0)
af.print_array(x1)

p = af.lu_inplace(a)

x2 = af.solve_lu(a, p, b)

af.print_array(x2)

print(af.rank(a))
print(af.det(a))
print(af.norm(a, af.AF_NORM_EUCLID))
print(af.norm(a, af.AF_NORM_MATRIX_1))
print(af.norm(a, af.AF_NORM_MATRIX_INF))
print(af.norm(a, af.AF_NORM_MATRIX_L_PQ, 1, 1))
