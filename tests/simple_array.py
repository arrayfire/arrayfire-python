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
import array as host

a = af.array([1, 2, 3])
af.print_array(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())


a = af.array(host.array('i', [4, 5, 6]))
af.print_array(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

a = af.array(host.array('l', [7, 8, 9] * 4), (2, 5))
af.print_array(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

a = af.randu(5, 5)
af.print_array(a)
b = af.array(a)
af.print_array(b)

c = a.copy()
af.print_array(c)
af.print_array(a[0,0])
af.print_array(a[0])
af.print_array(a[:])
af.print_array(a[:,:])
af.print_array(a[0:3,])
af.print_array(a[-2:-1,-1])
af.print_array(a[0:5])
af.print_array(a[0:5:2])

idx = af.array(host.array('i', [0, 3, 2]))
af.print_array(idx)
aa = a[idx]
af.print_array(aa)

a[0] = 1
af.print_array(a)
a[0] = af.randu(1, 5)
af.print_array(a)
a[:] = af.randu(5,5)
af.print_array(a)
a[:,-1] = af.randu(5,1)
af.print_array(a)
a[0:5:2] = af.randu(3, 5)
af.print_array(a)
a[idx, idx] = af.randu(3,3)
af.print_array(a)
