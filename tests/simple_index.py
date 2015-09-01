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
from arrayfire import ParallelRange
import array as host

a = af.randu(5, 5)
af.display(a)
b = af.Array(a)
af.display(b)

c = a.copy()
af.display(c)
af.display(a[0,0])
af.display(a[0])
af.display(a[:])
af.display(a[:,:])
af.display(a[0:3,])
af.display(a[-2:-1,-1])
af.display(a[0:5])
af.display(a[0:5:2])

idx = af.Array(host.array('i', [0, 3, 2]))
af.display(idx)
aa = a[idx]
af.display(aa)

a[0] = 1
af.display(a)
a[0] = af.randu(1, 5)
af.display(a)
a[:] = af.randu(5,5)
af.display(a)
a[:,-1] = af.randu(5,1)
af.display(a)
a[0:5:2] = af.randu(3, 5)
af.display(a)
a[idx, idx] = af.randu(3,3)
af.display(a)


a = af.randu(5,1)
b = af.randu(5,1)
af.display(a)
af.display(b)
for ii in ParallelRange(1,3):
    a[ii] = b[ii]

af.display(a)

for ii in ParallelRange(2,5):
    b[ii] = 2
af.display(b)

a = af.randu(3,2)
rows = af.constant(0, 1, dtype=af.Dtype.s32)
b = a[:,rows]
af.display(b)
for r in rows:
    af.display(r)
    af.display(b[:,r])
