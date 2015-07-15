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

a = af.randu(3, 3)

print(af.sum(a), af.product(a), af.min(a), af.max(a), af.count(a), af.any_true(a), af.all_true(a))

af.print_array(af.sum(a, 0))
af.print_array(af.sum(a, 1))

af.print_array(af.product(a, 0))
af.print_array(af.product(a, 1))

af.print_array(af.min(a, 0))
af.print_array(af.min(a, 1))

af.print_array(af.max(a, 0))
af.print_array(af.max(a, 1))

af.print_array(af.count(a, 0))
af.print_array(af.count(a, 1))

af.print_array(af.any_true(a, 0))
af.print_array(af.any_true(a, 1))

af.print_array(af.all_true(a, 0))
af.print_array(af.all_true(a, 1))

af.print_array(af.accum(a, 0))
af.print_array(af.accum(a, 1))

af.print_array(af.sort(a, is_ascending=True))
af.print_array(af.sort(a, is_ascending=False))

val,idx = af.sort_index(a, is_ascending=True)
af.print_array(val)
af.print_array(idx)
val,idx = af.sort_index(a, is_ascending=False)
af.print_array(val)
af.print_array(idx)

b = af.randu(3,3)
keys,vals = af.sort_by_key(a, b, is_ascending=True)
af.print_array(keys)
af.print_array(vals)
keys,vals = af.sort_by_key(a, b, is_ascending=False)
af.print_array(keys)
af.print_array(vals)

c = af.randu(5,1)
d = af.randu(5,1)
cc = af.set_unique(c, is_sorted=False)
dd = af.set_unique(af.sort(d), is_sorted=True)
af.print_array(cc)
af.print_array(dd)

af.print_array(af.set_union(cc, dd, is_unique=True))
af.print_array(af.set_union(cc, dd, is_unique=False))

af.print_array(af.set_intersect(cc, cc, is_unique=True))
af.print_array(af.set_intersect(cc, cc, is_unique=False))
