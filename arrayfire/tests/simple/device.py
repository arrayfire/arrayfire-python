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

    curr_dev = af.get_device()
    print_func(curr_dev)
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

    af.set_device(curr_dev)

    a = af.randu(10,10)
    display_func(a)
    dev_ptr = af.get_device_ptr(a)
    print_func(dev_ptr)
    b = af.Array(src=dev_ptr, dims=a.dims(), dtype=a.dtype(), is_device=True)
    display_func(b)

    c = af.randu(10,10)
    af.lock_array(c)
    af.unlock_array(c)

    a = af.constant(1, 3, 3)
    b = af.constant(2, 3, 3)
    af.eval(a)
    af.eval(b)
    print_func(a)
    print_func(b)
    c = a + b
    d = a - b
    af.eval(c, d)
    print_func(c)
    print_func(d)

    print_func(af.set_manual_eval_flag(True))
    assert(af.get_manual_eval_flag() == True)
    print_func(af.set_manual_eval_flag(False))
    assert(af.get_manual_eval_flag() == False)

    display_func(af.is_locked_array(a))

_util.tests['device'] = simple_device
