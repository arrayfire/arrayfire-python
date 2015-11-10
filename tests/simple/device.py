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

def simple_device(verbose=False):
    display_func = _util.display_func(verbose)
    print_func   = _util.print_func(verbose)
    print_func(af.device_info())
    print_func(af.get_device_count())
    print_func(af.is_dbl_supported())
    af.sync()

    dev = af.get_device()
    print_func(dev)
    for k in range(af.get_device_count()):
        af.set_device(k)
        dev = af.get_device()
        assert(k == dev)

        print_func(af.is_dbl_supported(k))

        af.device_gc()

        mem_info_old = af.device_mem_info()

        a = af.randu(100, 100)
        af.sync(dev)
        mem_info = af.device_mem_info()
        assert(mem_info['alloc']['buffers'] == 1 + mem_info_old['alloc']['buffers'])
        assert(mem_info[ 'lock']['buffers'] == 1 + mem_info_old[ 'lock']['buffers'])

    af.set_device(dev)

    a = af.randu(10,10)
    display_func(a)
    dev_ptr = af.get_device_ptr(a)
    print_func(dev_ptr)
    b = af.Array(src=dev_ptr, dims=a.dims(), dtype=a.dtype(), is_device=True)
    display_func(b)
    af.lock_device_ptr(b)
    af.unlock_device_ptr(b)

_util.tests['device'] = simple_device
