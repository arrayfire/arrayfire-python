#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from .library import *
from ctypes import *
from .util import (safe_call, to_str)

def info():
    safe_call(clib.af_info())

def device_info():
    c_char_256 = c_char * 256
    device_name = c_char_256()
    backend_name = c_char_256()
    toolkit = c_char_256()
    compute = c_char_256()

    safe_call(clib.af_device_info(pointer(device_name), pointer(backend_name), \
                                  pointer(toolkit), pointer(compute)))
    dev_info = {}
    dev_info['device'] = to_str(device_name)
    dev_info['backend'] = to_str(backend_name)
    dev_info['toolkit'] = to_str(toolkit)
    dev_info['compute'] = to_str(compute)

    return dev_info

def get_device_count():
    c_num = c_int(0)
    safe_call(clib.af_get_device_count(pointer(c_num)))
    return c_num.value

def get_device():
    c_dev = c_int(0)
    safe_call(clib.af_get_device(pointer(c_dev)))
    return c_dev.value

def set_device(num):
    safe_call(clib.af_set_device(num))

def is_dbl_supported(device=None):
    dev = device if device is not None else get_device()
    res = c_bool(False)
    safe_call(clib.af_get_dbl_support(pointer(res), dev))
    return res.value

def sync(device=None):
    dev = device if device is not None else get_device()
    safe_call(clib.af_sync(dev))

def device_mem_info():
    alloc_bytes = c_size_t(0)
    alloc_buffers = c_size_t(0)
    lock_bytes = c_size_t(0)
    lock_buffers = c_size_t(0)
    safe_call(clib.af_device_mem_info(pointer(alloc_bytes), pointer(alloc_buffers),\
                                      pointer(lock_bytes), pointer(lock_buffers)))
    mem_info = {}
    mem_info['alloc'] = {'buffers' : alloc_buffers.value, 'bytes' : alloc_bytes.value}
    mem_info['lock'] = {'buffers' : lock_buffers.value, 'bytes' : lock_bytes.value}
    return mem_info
