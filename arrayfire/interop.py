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

This module provides helper functions to copy data to arrayfire from the following modules:

     1. numpy - numpy.ndarray
     2. pycuda - pycuda.gpuarray
     3. pyopencl - pyopencl.array
     4. numba - numba.cuda.cudadrv.devicearray.DeviceNDArray

"""

from .array import *
from .device import *


def _fc_to_af_array(in_ptr, in_shape, in_dtype, is_device=False, copy = True):
    """
    Fortran Contiguous to af array
    """
    res = Array(in_ptr, in_shape, in_dtype, is_device=is_device)

    if not is_device:
        return res

    lock_array(res)
    return res.copy() if copy else res

def _cc_to_af_array(in_ptr, ndim, in_shape, in_dtype, is_device=False, copy = True):
    """
    C Contiguous to af array
    """
    if ndim == 1:
        return _fc_to_af_array(in_ptr, in_shape, in_dtype, is_device, copy)
    else:
        shape = tuple(reversed(in_shape))
        res = Array(in_ptr, shape, in_dtype, is_device=is_device)
        if is_device: lock_array(res)
        return res._reorder()

_nptype_to_aftype = {'b1' : Dtype.b8,
		     'u1' : Dtype.u8,
		     'u2' : Dtype.u16,
		     'i2' : Dtype.s16,
		     's4' : Dtype.u32,
		     'i4' : Dtype.s32,
		     'f4' : Dtype.f32,
		     'c8' : Dtype.c32,
		     's8' : Dtype.u64,
		     'i8' : Dtype.s64,
                     'f8' : Dtype.f64,
                     'c16' : Dtype.c64}

try:
    import numpy as np
except ImportError:
    AF_NUMPY_FOUND=False
else:
    from numpy import ndarray as NumpyArray
    from .data import reorder

    AF_NUMPY_FOUND=True

    def np_to_af_array(np_arr, copy=True):
        """
        Convert numpy.ndarray to arrayfire.Array.

        Parameters
        ----------
        np_arr  : numpy.ndarray()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

        Returns
        ---------
        af_arr  : arrayfire.Array()
        """

        in_shape = np_arr.shape
        in_ptr = np_arr.ctypes.data_as(c_void_ptr_t)
        in_dtype = _nptype_to_aftype[np_arr.dtype.str[1:]]

        if not copy:
            raise RuntimeError("Copy can not be False for numpy arrays")

        if (np_arr.flags['F_CONTIGUOUS']):
            return _fc_to_af_array(in_ptr, in_shape, in_dtype)
        elif (np_arr.flags['C_CONTIGUOUS']):
            return _cc_to_af_array(in_ptr, np_arr.ndim, in_shape, in_dtype)
        else:
            return np_to_af_array(np_arr.copy())

    from_ndarray = np_to_af_array

try:
    import pycuda.gpuarray
except ImportError:
    AF_PYCUDA_FOUND=False
else:
    from pycuda.gpuarray import GPUArray as CudaArray
    AF_PYCUDA_FOUND=True

    def pycuda_to_af_array(pycu_arr, copy=True):
        """
        Convert pycuda.gpuarray to arrayfire.Array

        Parameters
        -----------
        pycu_arr  : pycuda.GPUArray()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

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

        if not copy and not pycu_arr.flags.f_contiguous:
            raise RuntimeError("Copy can only be False when arr.flags.f_contiguous is True")

        if (pycu_arr.flags.f_contiguous):
            return _fc_to_af_array(in_ptr, in_shape, in_dtype, True, copy)
        elif (pycu_arr.flags.c_contiguous):
            return _cc_to_af_array(in_ptr, pycu_arr.ndim, in_shape, in_dtype, True, copy)
        else:
            return pycuda_to_af_array(pycu_arr.copy())

try:
    from pyopencl.array import Array as OpenclArray
except ImportError:
    AF_PYOPENCL_FOUND=False
else:
    from .opencl import add_device_context as _add_device_context
    from .opencl import set_device_context as _set_device_context
    from .opencl import get_device_id as _get_device_id
    from .opencl import get_context as _get_context
    AF_PYOPENCL_FOUND=True

    def pyopencl_to_af_array(pycl_arr, copy=True):
        """
        Convert pyopencl.gpuarray to arrayfire.Array

        Parameters
        -----------
        pycl_arr  : pyopencl.Array()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

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
            print("Adding context and queue")
            _add_device_context(dev, ctx, que)
            _set_device_context(dev, ctx)

        info()
        in_ptr = pycl_arr.base_data.int_ptr
        in_shape = pycl_arr.shape
        in_dtype = pycl_arr.dtype.char

        if not copy and not pycl_arr.flags.f_contiguous:
            raise RuntimeError("Copy can only be False when arr.flags.f_contiguous is True")

        print("Copying array")
        print(pycl_arr.base_data.int_ptr)
        if (pycl_arr.flags.f_contiguous):
            return _fc_to_af_array(in_ptr, in_shape, in_dtype, True, copy)
        elif (pycl_arr.flags.c_contiguous):
            return _cc_to_af_array(in_ptr, pycl_arr.ndim, in_shape, in_dtype, True, copy)
        else:
            return pyopencl_to_af_array(pycl_arr.copy())

try:
    import numba
except ImportError:
    AF_NUMBA_FOUND=False
else:
    from numba import cuda
    NumbaCudaArray = cuda.cudadrv.devicearray.DeviceNDArray
    AF_NUMBA_FOUND=True

    def numba_to_af_array(nb_arr, copy=True):
        """
        Convert numba.gpuarray to arrayfire.Array

        Parameters
        -----------
        nb_arr  : numba.cuda.cudadrv.devicearray.DeviceNDArray()

        copy : Bool specifying if array is to be copied.
               Default is true.
               Can only be False if array is fortran contiguous.

        Returns
        ----------
        af_arr    : arrayfire.Array()

        Note
        ----------
        The input array is copied to af.Array
        """

        in_ptr = nb_arr.device_ctypes_pointer.value
        in_shape = nb_arr.shape
        in_dtype = _nptype_to_aftype[nb_arr.dtype.str[1:]]

        if not copy and not nb_arr.flags.f_contiguous:
            raise RuntimeError("Copy can only be False when arr.flags.f_contiguous is True")

        if (nb_arr.is_f_contiguous()):
            return _fc_to_af_array(in_ptr, in_shape, in_dtype, True, copy)
        elif (nb_arr.is_c_contiguous()):
            return _cc_to_af_array(in_ptr, nb_arr.ndim, in_shape, in_dtype, True, copy)
        else:
            return numba_to_af_array(nb_arr.copy())

def to_array(in_array, copy = True):
    """
    Helper function to convert input from a different module to af.Array

    Parameters
    -------------

    in_array : array like object
             Can be one of the following:
             - numpy.ndarray
             - pycuda.GPUArray
             - pyopencl.Array
             - numba.cuda.cudadrv.devicearray.DeviceNDArray
             - array.array
             - list
    copy : Bool specifying if array is to be copied.
          Default is true.
          Can only be False if array is fortran contiguous.

    Returns
    --------------
    af.Array of same dimensions as input after copying the data from the input

    """
    if AF_NUMPY_FOUND and isinstance(in_array, NumpyArray):
        return np_to_af_array(in_array, copy)
    if AF_PYCUDA_FOUND and isinstance(in_array, CudaArray):
        return pycuda_to_af_array(in_array, copy)
    if AF_PYOPENCL_FOUND and isinstance(in_array, OpenclArray):
        return pyopencl_to_af_array(in_array, copy)
    if AF_NUMBA_FOUND and isinstance(in_array, NumbaCudaArray):
        return numba_to_af_array(in_array, copy)
    return Array(src=in_array)
