#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Functions specific to OpenCL backend.

This module provides interoperability with other OpenCL libraries.
"""

from .util import *
from .library import (_Enum, _Enum_Type)

class DEVICE_TYPE(_Enum):
    """
    ArrayFire wrapper for CL_DEVICE_TYPE
    """
    CPU = _Enum_Type(1<<1)
    GPU = _Enum_Type(1<<2)
    ACC = _Enum_Type(1<<3)
    UNKNOWN = _Enum_Type(-1)

class PLATFORM(_Enum):
    """
    ArrayFire enum for common platforms
    """
    AMD     = _Enum_Type(0)
    APPLE   = _Enum_Type(1)
    INTEL   = _Enum_Type(2)
    NVIDIA  = _Enum_Type(3)
    BEIGNET = _Enum_Type(4)
    POCL    = _Enum_Type(5)
    UNKNOWN = _Enum_Type(-1)

def get_context(retain=False):
    """
    Get the current OpenCL context being used by ArrayFire.

    Parameters
    ----------

    retain : bool. optional. Default: False.
        Specifies if the context needs to be retained by arrayfire before returning.

    Returns
    -----------
    context : integer denoting the context id.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    context = c_void_ptr_t(0)
    safe_call(backend.get().afcl_get_context(c_pointer(context), retain))
    return context.value

def get_queue(retain):
    """
    Get the current OpenCL command queue being used by ArrayFire.

    Parameters
    ----------

    retain : bool. optional. Default: False.
        Specifies if the context needs to be retained by arrayfire before returning.

    Returns
    -----------
    queue : integer denoting the queue id.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    queue = c_int_t(0)
    safe_call(backend.get().afcl_get_queue(c_pointer(queue), retain))
    return queue.value

def get_device_id():
    """
    Get native (unsorted) OpenCL device ID

    Returns
    --------

    idx : int.
        Specifies the `cl_device_id` of the device.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    idx = c_int_t(0)
    safe_call(backend.get().afcl_get_device_id(c_pointer(idx)))
    return idx.value

def set_device_id(idx):
    """
    Set native (unsorted) OpenCL device ID

    Parameters
    ----------

    idx : int.
        Specifies the `cl_device_id` of the device.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcl_set_device_id(idx))
    return

def add_device_context(dev, ctx, que):
    """
    Add a new device to arrayfire opencl device manager

    Parameters
    ----------

    dev : cl_device_id

    ctx : cl_context

    que : cl_command_queue

    """
    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcl_add_device_context(dev, ctx, que))

def set_device_context(dev, ctx):
    """
    Set a device as current active device

    Parameters
    ----------

    dev  : cl_device_id

    ctx  : cl_context

    """
    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcl_set_device_context(dev, ctx))

def delete_device_context(dev, ctx):
    """
    Delete a device

    Parameters
    ----------

    dev  : cl_device_id

    ctx  : cl_context

    """
    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcl_delete_device_context(dev, ctx))


_to_device_type = {DEVICE_TYPE.CPU.value     : DEVICE_TYPE.CPU,
                   DEVICE_TYPE.GPU.value     : DEVICE_TYPE.GPU,
                   DEVICE_TYPE.ACC.value     : DEVICE_TYPE.ACC,
                   DEVICE_TYPE.UNKNOWN.value : DEVICE_TYPE.UNKNOWN}

_to_platform    = {PLATFORM.AMD.value     : PLATFORM.AMD,
                   PLATFORM.APPLE.value   : PLATFORM.APPLE,
                   PLATFORM.INTEL.value   : PLATFORM.INTEL,
                   PLATFORM.NVIDIA.value  : PLATFORM.NVIDIA,
                   PLATFORM.BEIGNET.value : PLATFORM.BEIGNET,
                   PLATFORM.POCL.value    : PLATFORM.POCL,
                   PLATFORM.UNKNOWN.value : PLATFORM.UNKNOWN}


def get_device_type():
    """
    Get opencl device type
    """
    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    res = c_int_t(DEVICE_TYPE.UNKNOWN.value)
    safe_call(backend.get().afcl_get_device_type(c_pointer(res)))
    return _to_device_type[res.value]

def get_platform():
    """
    Get opencl platform
    """
    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    res = c_int_t(PLATFORM.UNKNOWN.value)
    safe_call(backend.get().afcl_get_platform(c_pointer(res)))
    return _to_platform[res.value]
