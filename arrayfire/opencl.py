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
    from .library import backend as backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    context = ct.c_void_p(0)
    safe_call(backend.get().afcl_get_context(ct.pointer(context), retain))
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
    from .library import backend as backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    queue = ct.c_int(0)
    safe_call(backend.get().afcl_get_queue(ct.pointer(queue), retain))
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
    from .library import backend as backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    idx = ct.c_int(0)
    safe_call(backend.get().afcl_get_device_id(ct.pointer(idx)))
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
    from .library import backend as backend

    if (backend.name() != "opencl"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcl_set_device_id(idx))
    return
