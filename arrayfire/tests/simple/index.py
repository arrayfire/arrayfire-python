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
from . import _util

def simple_index(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)
    a = af.randu(5, 5)
    display_func(a)
    b = af.Array(a)
    display_func(b)

    c = a.copy()
    display_func(c)
    display_func(a[0,0])
    display_func(a[0])
    display_func(a[:])
    display_func(a[:,:])
    display_func(a[0:3,])
    display_func(a[-2:-1,-1])
    display_func(a[0:5])
    display_func(a[0:5:2])

    idx = af.Array(host.array('i', [0, 3, 2]))
    display_func(idx)
    aa = a[idx]
    display_func(aa)

    a[0] = 1
    display_func(a)
    a[0] = af.randu(1, 5)
    display_func(a)
    a[:] = af.randu(5,5)
    display_func(a)
    a[:,-1] = af.randu(5,1)
    display_func(a)
    a[0:5:2] = af.randu(3, 5)
    display_func(a)
    a[idx, idx] = af.randu(3,3)
    display_func(a)


    a = af.randu(5,1)
    b = af.randu(5,1)
    display_func(a)
    display_func(b)
    for ii in ParallelRange(1,3):
        a[ii] = b[ii]

    display_func(a)

    for ii in ParallelRange(2,5):
        b[ii] = 2
    display_func(b)

    a = af.randu(3,2)
    rows = af.constant(0, 1, dtype=af.Dtype.s32)
    b = a[:,rows]
    display_func(b)
    for r in rows:
        display_func(r)
        display_func(b[:,r])

    a = af.randu(3)
    c = af.randu(3)
    b = af.constant(1,3,dtype=af.Dtype.b8)
    display_func(a)
    a[b] = c
    display_func(a)

_util.tests['index'] = simple_index
