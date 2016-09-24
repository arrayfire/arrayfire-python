#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Functions specific to CUDA backend.

This module provides interoperability with other CUDA libraries.
"""

def get_stream(idx):
    """
    Get the CUDA stream used for the device `idx` by ArrayFire.

    Parameters
    ----------

    idx : int.
        Specifies the index of the device.

    Returns
    -----------
    stream : integer denoting the stream id.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend as backend

    if (backend.name() != "cuda"):
        raise RuntimeError("Invalid backend loaded")

    stream = c_void_ptr_t(0)
    safe_call(backend.get().afcu_get_stream(c_pointer(stream), idx))
    return stream.value

def get_native_id(idx):
    """
    Get native (unsorted) CUDA device ID

    Parameters
    ----------

    idx : int.
        Specifies the (sorted) index of the device.

    Returns
    -----------
    native_idx : integer denoting the native cuda id.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend as backend

    if (backend.name() != "cuda"):
        raise RuntimeError("Invalid backend loaded")

    native = c_int_t(0)
    safe_call(backend.get().afcu_get_native_id(c_pointer(native), idx))
    return native.value

def set_native_id(idx):
    """
    Set native (unsorted) CUDA device ID

    Parameters
    ----------

    idx : int.
        Specifies the (unsorted) native index of the device.
    """

    import ctypes as ct
    from .util import safe_call as safe_call
    from .library import backend as backend

    if (backend.name() != "cuda"):
        raise RuntimeError("Invalid backend loaded")

    safe_call(backend.get().afcu_set_native_id(idx))
    return
