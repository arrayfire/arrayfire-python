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
from arrayfire import parallel_range
import array as host

a = af.randu(5, 5)
af.display(a)
b = af.array(a)
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

idx = af.array(host.array('i', [0, 3, 2]))
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
for ii in parallel_range(1,3):
    a[ii] = b[ii]

af.display(a)

for ii in parallel_range(2,5):
    b[ii] = 2
af.display(b)
