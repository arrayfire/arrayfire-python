#!/usr/bin/env python

#######################################################
# Copyright (c) 2020, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af


def test_simple_device() -> None:
    assert af.device_info()
    assert af.get_device_count()
    assert af.is_dbl_supported()
    af.sync()

    curr_dev = af.get_device()
    for k in range(af.get_device_count()):
        af.set_device(k)
        dev = af.get_device()
        assert k == dev

        assert af.is_dbl_supported(k)

        af.device_gc()

        mem_info_old = af.device_mem_info()

        a = af.randu(100, 100)
        af.sync(dev)
        mem_info = af.device_mem_info()
        assert mem_info["alloc"]["buffers"] == 1 + mem_info_old["alloc"]["buffers"]
        assert mem_info["lock"]["buffers"] == 1 + mem_info_old["lock"]["buffers"]

    af.set_device(curr_dev)

    a = af.randu(10, 10)
    assert a
    dev_ptr = af.get_device_ptr(a)
    assert dev_ptr
    b = af.Array(src=dev_ptr, dims=a.dims(), dtype=a.dtype(), is_device=True)
    assert b

    c = af.randu(10, 10)
    af.lock_array(c)
    af.unlock_array(c)

    a = af.constant(1, 3, 3)
    b = af.constant(2, 3, 3)
    af.eval(a)
    af.eval(b)
    assert a
    assert b
    c = a + b
    d = a - b
    af.eval(c, d)
    assert c
    assert d

    assert not af.set_manual_eval_flag(True)
    assert af.get_manual_eval_flag()
    assert not af.set_manual_eval_flag(False)
    assert not af.get_manual_eval_flag()

    assert not af.is_locked_array(a)
