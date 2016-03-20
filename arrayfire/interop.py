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
from .device import lock_array

try:
    import numpy as np
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
        if (np_arr.flags['F_CONTIGUOUS']):
            return Array(np_arr.ctypes.data, np_arr.shape, np_arr.dtype.char)
        elif (np_arr.flags['C_CONTIGUOUS']):
            if np_arr.ndim == 1:
                return Array(np_arr.ctypes.data, np_arr.shape, np_arr.dtype.char)
            elif np_arr.ndim == 2:
                shape = (np_arr.shape[1], np_arr.shape[0])
                res = Array(np_arr.ctypes.data, shape, np_arr.dtype.char)
                return reorder(res, 1, 0)
            elif np_arr.ndim == 3:
                shape = (np_arr.shape[2], np_arr.shape[1], np_arr.shape[0])
                res = Array(np_arr.ctypes.data, shape, np_arr.dtype.char)
                return reorder(res, 2, 1, 0)
            elif np_arr.ndim == 4:
                shape = (np_arr.shape[3], np_arr.shape[2], np_arr.shape[1], np_arr.shape[0])
                res = Array(np_arr.ctypes.data, shape, np_arr.dtype.char)
                return reorder(res, 3, 2, 1, 0)
            else:
                raise RuntimeError("Unsupported ndim")
        else:
            return np_to_af_array(np.asfortranarray(np_arr))

    from_ndarray = np_to_af_array
except:
    AF_NUMPY_FOUND=False

try:
    import pycuda.gpuarray as CudaArray
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
        """
        if (pycu_arr.flags.f_contiguous):
            res = Array(pycu_arr.ptr, pycu_arr.shape, pycu_arr.dtype.char, is_device=True)
            lock_array(res)
            return res
        elif (pycu_arr.flags.c_contiguous):
            if pycu_arr.ndim == 1:
                return Array(pycu_arr.ptr, pycu_arr.shape, pycu_arr.dtype.char, is_device=True)
            elif pycu_arr.ndim == 2:
                shape = (pycu_arr.shape[1], pycu_arr.shape[0])
                res = Array(pycu_arr.ptr, shape, pycu_arr.dtype.char, is_device=True)
                lock_array(res)
                return reorder(res, 1, 0)
            elif pycu_arr.ndim == 3:
                shape = (pycu_arr.shape[2], pycu_arr.shape[1], pycu_arr.shape[0])
                res = Array(pycu_arr.ptr, shape, pycu_arr.dtype.char, is_device=True)
                lock_array(res)
                return reorder(res, 2, 1, 0)
            elif pycu_arr.ndim == 4:
                shape = (pycu_arr.shape[3], pycu_arr.shape[2], pycu_arr.shape[1], pycu_arr.shape[0])
                res = Array(pycu_arr.ptr, shape, pycu_arr.dtype.char, is_device=True)
                lock_array(res)
                return reorder(res, 3, 2, 1, 0)
            else:
                raise RuntimeError("Unsupported ndim")
        else:
            return pycuda_to_af_array(pycu_arr.copy())
except:
    AF_PYCUDA_FOUND=False
