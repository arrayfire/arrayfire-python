#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Interop with other python packages.

This module provides interoperability with the following python packages.

     1. numpy
     2. pycuda
"""

from .array import *
from .device import *

try:
    import numpy as np
    from numpy import ndarray as NumpyArray
    from .data import reorder

    AF_NUMPY_FOUND=True

    def np_to_af_array(np_arr):
        """
        Convert numpy.ndarray to arrayfire.Array.

        Parameters
        ----------
        np_arr  : numpy.ndarray()

        Returns
        ---------
        af_arr  : arrayfire.Array()
        """

        in_shape = np_arr.shape
        in_ptr = np_arr.ctypes.data
        in_dtype = np_arr.dtype.char

        if (np_arr.flags['F_CONTIGUOUS']):
            return Array(in_ptr, in_shape, in_dtype)
        elif (np_arr.flags['C_CONTIGUOUS']):
            if np_arr.ndim == 1:
                return Array(in_ptr, in_shape, in_dtype)
            elif np_arr.ndim == 2:
                shape = (in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype)
                return reorder(res, 1, 0)
            elif np_arr.ndim == 3:
                shape = (in_shape[2], in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype)
                return reorder(res, 2, 1, 0)
            elif np_arr.ndim == 4:
                shape = (in_shape[3], in_shape[2], in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype)
                return reorder(res, 3, 2, 1, 0)
            else:
                raise RuntimeError("Unsupported ndim")
        else:
            return np_to_af_array(np.asfortranarray(np_arr))

    from_ndarray = np_to_af_array
except:
    AF_NUMPY_FOUND=False

try:
    import pycuda.gpuarray
    from pycuda.gpuarray import GPUArray as CudaArray
    AF_PYCUDA_FOUND=True

    def pycuda_to_af_array(pycu_arr):
        """
        Convert pycuda.gpuarray to arrayfire.Array

        Parameters
        -----------
        pycu_arr  : pycuda.GPUArray()

        Returns
        ----------
        af_arr    : arrayfire.Array()

        Note
        ----------
        The input array is copied to af.Array
        """

        in_ptr = pycu_arr.ptr
        in_shape = pycu_arr.shape
        in_dtype = pycu_arr.dtype.char

        if (pycu_arr.flags.f_contiguous):
            res = Array(in_ptr, in_shape, in_dtype, is_device=True)
            lock_array(res)
            res = res.copy()
            return res
        elif (pycu_arr.flags.c_contiguous):
            if pycu_arr.ndim == 1:
                return Array(in_ptr, in_shape, in_dtype, is_device=True)
            elif pycu_arr.ndim == 2:
                shape = (in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype, is_device=True)
                lock_array(res)
                return reorder(res, 1, 0)
            elif pycu_arr.ndim == 3:
                shape = (in_shape[2], in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype, is_device=True)
                lock_array(res)
                return reorder(res, 2, 1, 0)
            elif pycu_arr.ndim == 4:
                shape = (in_shape[3], in_shape[2], in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype, is_device=True)
                lock_array(res)
                return reorder(res, 3, 2, 1, 0)
            else:
                raise RuntimeError("Unsupported ndim")
        else:
            return pycuda_to_af_array(pycu_arr.copy())
except:
    AF_PYCUDA_FOUND=False

try:
    from pyopencl.array import Array as OpenclArray
    from .opencl import add_device_context as _add_device_context
    from .opencl import set_device_context as _set_device_context
    from .opencl import get_device_id as _get_device_id
    from .opencl import get_context as _get_context
    AF_PYOPENCL_FOUND=True

    def pyopencl_to_af_array(pycl_arr):
        """
        Convert pyopencl.gpuarray to arrayfire.Array

        Parameters
        -----------
        pycl_arr  : pyopencl.Array()

        Returns
        ----------
        af_arr    : arrayfire.Array()

        Note
        ----------
        The input array is copied to af.Array
        """

        ctx = pycl_arr.context.int_ptr
        que = pycl_arr.queue.int_ptr
        dev = pycl_arr.queue.device.int_ptr

        dev_idx = None
        ctx_idx = None
        for n in range(get_device_count()):
            set_device(n)
            dev_idx = _get_device_id()
            ctx_idx = _get_context()
            if (dev_idx == dev and ctx_idx == ctx):
                break

        if (dev_idx == None or ctx_idx == None or
            dev_idx != dev or ctx_idx != ctx):
            _add_device_context(dev, ctx, que)
            _set_device_context(dev, ctx)

        in_ptr = pycl_arr.base_data.int_ptr
        in_shape = pycl_arr.shape
        in_dtype = pycl_arr.dtype.char

        if (pycl_arr.flags.f_contiguous):
            res = Array(in_ptr, in_shape, in_dtype, is_device=True)
            lock_array(res)
            return res
        elif (pycl_arr.flags.c_contiguous):
            if pycl_arr.ndim == 1:
                return Array(in_ptr, in_shape, in_dtype, is_device=True)
            elif pycl_arr.ndim == 2:
                shape = (in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype, is_device=True)
                lock_array(res)
                return reorder(res, 1, 0)
            elif pycl_arr.ndim == 3:
                shape = (in_shape[2], in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype, is_device=True)
                lock_array(res)
                return reorder(res, 2, 1, 0)
            elif pycl_arr.ndim == 4:
                shape = (in_shape[3], in_shape[2], in_shape[1], in_shape[0])
                res = Array(in_ptr, shape, in_dtype, is_device=True)
                lock_array(res)
                return reorder(res, 3, 2, 1, 0)
            else:
                raise RuntimeError("Unsupported ndim")
        else:
            return pyopencl_to_af_array(pycl_arr.copy())
except:
    AF_PYOPENCL_FOUND=False


def to_array(in_array):
    """
    Helper function to convert input from a different module to af.Array

    Parameters
    -------------

    in_array : array like object
             Can be one of numpy.ndarray, pycuda.GPUArray, pyopencl.Array, array.array, list

    Returns
    --------------
    af.Array of same dimensions as input after copying the data from the input


    """
    if AF_NUMPY_FOUND and isinstance(in_array, NumpyArray):
        return np_to_af_array(in_array)
    if AF_PYCUDA_FOUND and isinstance(in_array, CudaArray):
        return pycuda_to_af_array(in_array)
    if AF_PYOPENCL_FOUND and isinstance(in_array, OpenclArray):
        return pyopencl_to_af_array(in_array)
    return Array(src=in_array)
