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
af.display(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())


a = af.array(host.array('i', [4, 5, 6]))
af.display(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

a = af.array(host.array('l', [7, 8, 9] * 3), (3,3))
af.display(a)
print(a.elements(), a.type(), a.dims(), a.numdims())
print(a.is_empty(), a.is_scalar(), a.is_column(), a.is_row())
print(a.is_complex(), a.is_real(), a.is_double(), a.is_single())
print(a.is_real_floating(), a.is_floating(), a.is_integer(), a.is_bool())

af.display(af.transpose(a))

af.transpose_inplace(a)
af.display(a)

c = a.to_ctype()
for n in range(a.elements()):
    print(c[n])

c,s = a.to_ctype(True, True)
for n in range(a.elements()):
    print(c[n])
print(s)

arr = a.to_array()
lst = a.to_list(True)

print(arr)
print(lst)
