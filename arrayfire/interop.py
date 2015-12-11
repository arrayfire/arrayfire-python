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
"""

from .array import *

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
