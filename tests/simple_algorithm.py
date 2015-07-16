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

a = af.randu(3, 3)

print(af.sum(a), af.product(a), af.min(a), af.max(a), af.count(a), af.any_true(a), af.all_true(a))

af.display(af.sum(a, 0))
af.display(af.sum(a, 1))

af.display(af.product(a, 0))
af.display(af.product(a, 1))

af.display(af.min(a, 0))
af.display(af.min(a, 1))

af.display(af.max(a, 0))
af.display(af.max(a, 1))

af.display(af.count(a, 0))
af.display(af.count(a, 1))

af.display(af.any_true(a, 0))
af.display(af.any_true(a, 1))

af.display(af.all_true(a, 0))
af.display(af.all_true(a, 1))

af.display(af.accum(a, 0))
af.display(af.accum(a, 1))

af.display(af.sort(a, is_ascending=True))
af.display(af.sort(a, is_ascending=False))

val,idx = af.sort_index(a, is_ascending=True)
af.display(val)
af.display(idx)
val,idx = af.sort_index(a, is_ascending=False)
af.display(val)
af.display(idx)

b = af.randu(3,3)
keys,vals = af.sort_by_key(a, b, is_ascending=True)
af.display(keys)
af.display(vals)
keys,vals = af.sort_by_key(a, b, is_ascending=False)
af.display(keys)
af.display(vals)

c = af.randu(5,1)
d = af.randu(5,1)
cc = af.set_unique(c, is_sorted=False)
dd = af.set_unique(af.sort(d), is_sorted=True)
af.display(cc)
af.display(dd)

af.display(af.set_union(cc, dd, is_unique=True))
af.display(af.set_union(cc, dd, is_unique=False))

af.display(af.set_intersect(cc, cc, is_unique=True))
af.display(af.set_intersect(cc, cc, is_unique=False))
