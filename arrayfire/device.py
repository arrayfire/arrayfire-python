#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
"""
Functions to handle the available devices in the backend.
"""

from .library import *
from .util import (safe_call, to_str, get_version)

def init():
    """
    Note
    -----
    This function may need to be called when interoperating with other libraries
    """
    safe_call(backend.get().af_init())

def info():
    """
    Displays the information about the following:
        - ArrayFire build and version number.
        - The number of devices available.
        - The names of the devices.
        - The current device being used.
    """
    safe_call(backend.get().af_info())

def device_info():
    """
    Returns a map with the following fields:
        - 'device': Name of the current device.
        - 'backend': The current backend being used.
        - 'toolkit': The toolkit version for the backend.
        - 'compute': The compute version of the device.
    """
    c_char_256 = ct.c_char * 256
    device_name = c_char_256()
    backend_name = c_char_256()
    toolkit = c_char_256()
    compute = c_char_256()

    safe_call(backend.get().af_device_info(ct.pointer(device_name), ct.pointer(backend_name),
                                           ct.pointer(toolkit), ct.pointer(compute)))
    dev_info = {}
    dev_info['device'] = to_str(device_name)
    dev_info['backend'] = to_str(backend_name)
    dev_info['toolkit'] = to_str(toolkit)
    dev_info['compute'] = to_str(compute)

    return dev_info

def get_device_count():
    """
    Returns the number of devices available.
    """
    c_num = ct.c_int(0)
    safe_call(backend.get().af_get_device_count(ct.pointer(c_num)))
    return c_num.value

def get_device():
    """
    Returns the id of the current device.
    """
    c_dev = ct.c_int(0)
    safe_call(backend.get().af_get_device(ct.pointer(c_dev)))
    return c_dev.value

def set_device(num):
    """
    Change the active device to the specified id.

    Parameters
    ---------
    num: int.
         id of the desired device.
    """
    safe_call(backend.get().af_set_device(num))

def info_str(verbose = False):
    """
    Returns information about the following as a string:
        - ArrayFire version number.
        - The number of devices available.
        - The names of the devices.
        - The current device being used.
    """
    import platform
    res_str = 'ArrayFire'

    major, minor, patch = get_version()
    dev_info = device_info()
    backend_str = dev_info['backend']

    res_str += ' v' + str(major) + '.' + str(minor) + '.' + str(patch)
    res_str += ' (' + backend_str + ' ' + platform.architecture()[0] + ')\n'

    num_devices = get_device_count()
    curr_device_id = get_device()

    for n in range(num_devices):
        # To suppress warnings on CPU
        if (n != curr_device_id):
            set_device(n)

        if (n == curr_device_id):
            res_str += '[%d] ' % n
        else:
            res_str += '-%d- ' % n

        dev_info = device_info()

        if (backend_str.lower() == 'opencl'):
            res_str += dev_info['toolkit']

        res_str += ': ' + dev_info['device']

        if (backend_str.lower() != 'cpu'):
            res_str += ' (Compute ' + dev_info['compute'] + ')'

        res_str += '\n'

    # To suppress warnings on CPU
    if (curr_device_id != get_device()):
        set_device(curr_device_id)

    return res_str

def is_dbl_supported(device=None):
    """
    Check if double precision is supported on specified device.

    Parameters
    ---------
    device: optional: int. default: None.
         id of the desired device.

    Returns
    --------
        - True if double precision supported.
        - False if double precision not supported.
    """
    dev = device if device is not None else get_device()
    res = ct.c_bool(False)
    safe_call(backend.get().af_get_dbl_support(ct.pointer(res), dev))
    return res.value

def sync(device=None):
    """
    Block until all the functions on the device have completed execution.

    Parameters
    ---------
    device: optional: int. default: None.
         id of the desired device.
    """
    dev = device if device is not None else get_device()
    safe_call(backend.get().af_sync(dev))

def __eval(*args):
    for A in args:
        if isinstance(A, tuple):
            __eval(*A)
        if isinstance(A, list):
            __eval(*A)
        if isinstance(A, Array):
            safe_call(backend.get().af_eval(A.arr))

def eval(*args):
    """
    Evaluate the input

    Parameters
    -----------
    args : arguments to be evaluated
    """

    __eval(args)

def device_mem_info():
    """
    Returns a map with the following fields:
        - 'alloc': Contains the map of the following
            - 'buffers' : Total number of buffers allocated by memory manager.
            - 'bytes'   : Total number of bytes allocated by memory manager.
        - 'lock': Contains the map of the following
            - 'buffers' : Total number of buffers currently in scope.
            - 'bytes'   : Total number of bytes currently in scope.

    Note
    -----
    ArrayFire does not free memory when array goes out of scope. The memory is marked for reuse.
    - The difference between alloc buffers and lock buffers equals the number of free buffers.
    - The difference between alloc bytes and lock bytes equals the number of free bytes.

    """
    alloc_bytes = ct.c_size_t(0)
    alloc_buffers = ct.c_size_t(0)
    lock_bytes = ct.c_size_t(0)
    lock_buffers = ct.c_size_t(0)
    safe_call(backend.get().af_device_mem_info(ct.pointer(alloc_bytes), ct.pointer(alloc_buffers),
                                               ct.pointer(lock_bytes), ct.pointer(lock_buffers)))
    mem_info = {}
    mem_info['alloc'] = {'buffers' : alloc_buffers.value, 'bytes' : alloc_bytes.value}
    mem_info['lock'] = {'buffers' : lock_buffers.value, 'bytes' : lock_bytes.value}
    return mem_info

def device_gc():
    """
    Ask the garbage collector to free all unlocked memory
    """
    safe_call(backend.get().af_device_gc())

def get_device_ptr(a):
    """
    Get the raw device pointer of an array

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    Returns
    -------
        - internal device pointer held by a

    Note
    -----
        - The device pointer of `a` is not freed by memory manager until `unlock_device_ptr()` is called.
        - This function enables the user to interoperate arrayfire with other CUDA/OpenCL/C libraries.

    """
    ptr = ct.c_void_p(0)
    safe_call(backend.get().af_get_device_ptr(ct.pointer(ptr), a.arr))
    return ptr

def lock_device_ptr(a):
    """
    This functions is deprecated. Please use lock_array instead.
    """
    import warnings
    warnings.warn("This function is deprecated. Use lock_array instead.", DeprecationWarning)
    lock_array(a)

def lock_array(a):
    """
    Ask arrayfire to not perform garbage collection on raw data held by an array.

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    Note
    -----
        - The device pointer of `a` is not freed by memory manager until `unlock_device_ptr()` is called.
    """
    safe_call(backend.get().af_lock_array(a.arr))

def unlock_device_ptr(a):
    """
    This functions is deprecated. Please use unlock_array instead.
    """
    import warnings
    warnings.warn("This function is deprecated. Use unlock_array instead.", DeprecationWarning)
    unlock_array(a)

def unlock_array(a):
    """
    Tell arrayfire to resume garbage collection on raw data held by an array.

    Parameters
    ----------
    a: af.Array
       - A multi dimensional arrayfire array.

    """
    safe_call(backend.get().af_unlock_array(a.arr))

def alloc_device(num_bytes):
    """
    Allocate a buffer on the device with specified number of bytes.
    """
    ptr = ct.c_void_p(0)
    c_num_bytes = c_dim_t(num_bytes)
    safe_call(backend.get().af_alloc_device(ct.pointer(ptr), c_num_bytes))
    return ptr.value

def alloc_host(num_bytes):
    """
    Allocate a buffer on the host with specified number of bytes.
    """
    ptr = ct.c_void_p(0)
    c_num_bytes = c_dim_t(num_bytes)
    safe_call(backend.get().af_alloc_host(ct.pointer(ptr), c_num_bytes))
    return ptr.value

def alloc_pinned(num_bytes):
    """
    Allocate a buffer on the host using pinned memory with specified number of bytes.
    """
    ptr = ct.c_void_p(0)
    c_num_bytes = c_dim_t(num_bytes)
    safe_call(backend.get().af_alloc_pinned(ct.pointer(ptr), c_num_bytes))
    return ptr.value

def free_device(ptr):
    """
    Free the device memory allocated by alloc_device
    """
    cptr = ct.c_void_p(ptr)
    safe_call(backend.get().af_free_device(cptr))

def free_host(ptr):
    """
    Free the host memory allocated by alloc_host
    """
    cptr = ct.c_void_p(ptr)
    safe_call(backend.get().af_free_host(cptr))

def free_pinned(ptr):
    """
    Free the pinned memory allocated by alloc_pinned
    """
    cptr = ct.c_void_p(ptr)
    safe_call(backend.get().af_free_pinned(cptr))

from .array import Array
