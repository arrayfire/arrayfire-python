#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
BLAS functions (matmul, dot, etc)
"""

from .library import *
from .array import *

def matmul(lhs, rhs, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE):
    """
    Generalized matrix multiplication for two matrices.

    Parameters
    ----------

    lhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    rhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    lhs_opts: optional: af.MATPROP. default: af.MATPROP.NONE.
              Can be one of
               - af.MATPROP.NONE   - If no op should be done on `lhs`.
               - af.MATPROP.TRANS  - If `lhs` has to be transposed before multiplying.
               - af.MATPROP.CTRANS - If `lhs` has to be hermitian transposed before multiplying.

    rhs_opts: optional: af.MATPROP. default: af.MATPROP.NONE.
              Can be one of
               - af.MATPROP.NONE   - If no op should be done on `rhs`.
               - af.MATPROP.TRANS  - If `rhs` has to be transposed before multiplying.
               - af.MATPROP.CTRANS - If `rhs` has to be hermitian transposed before multiplying.

    Returns
    -------

    out : af.Array
          Output of the matrix multiplication on `lhs` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().af_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      lhs_opts.value, rhs_opts.value))
    return out

def matmulTN(lhs, rhs):
    """
    Matrix multiplication after transposing the first matrix.

    Parameters
    ----------

    lhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    rhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    Returns
    -------

    out : af.Array
          Output of the matrix multiplication on `transpose(lhs)` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().af_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.TRANS.value, MATPROP.NONE.value))
    return out

def matmulNT(lhs, rhs):
    """
    Matrix multiplication after transposing the second matrix.

    Parameters
    ----------

    lhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    rhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    Returns
    -------

    out : af.Array
          Output of the matrix multiplication on `lhs` and `transpose(rhs)`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().af_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.NONE.value, MATPROP.TRANS.value))
    return out

def matmulTT(lhs, rhs):
    """
    Matrix multiplication after transposing both inputs.

    Parameters
    ----------

    lhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    rhs : af.Array
          A 2 dimensional, real or complex arrayfire array.

    Returns
    -------

    out : af.Array
          Output of the matrix multiplication on `transpose(lhs)` and `transpose(rhs)`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    out = Array()
    safe_call(backend.get().af_matmul(c_pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.TRANS.value, MATPROP.TRANS.value))
    return out

def dot(lhs, rhs, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE, return_scalar = False):
    """
    Dot product of two input vectors.

    Parameters
    ----------

    lhs : af.Array
          A 1 dimensional, real or complex arrayfire array.

    rhs : af.Array
          A 1 dimensional, real or complex arrayfire array.

    lhs_opts: optional: af.MATPROP. default: af.MATPROP.NONE.
              Can be one of
               - af.MATPROP.NONE   - If no op should be done on `lhs`.
               - No other options are currently supported.

    rhs_opts: optional: af.MATPROP. default: af.MATPROP.NONE.
              Can be one of
               - af.MATPROP.NONE   - If no op should be done on `rhs`.
               - No other options are currently supported.

    return_scalar: optional: bool. default: False.
               - When set to true, the input arrays are flattened and the output is a scalar

    Returns
    -------

    out : af.Array or scalar
          Output of dot product of `lhs` and `rhs`.

    Note
    -----

    - The data types of `lhs` and `rhs` should be the same.
    - Batches are not supported.

    """
    if return_scalar:
        real = c_double_t(0)
        imag = c_double_t(0)
        safe_call(backend.get().af_dot_all(c_pointer(real), c_pointer(imag),
                                           lhs.arr, rhs.arr, lhs_opts.value, rhs_opts.value))
        real = real.value
        imag = imag.value
        return real if imag == 0 else real + imag * 1j
    else:
        out = Array()
        safe_call(backend.get().af_dot(c_pointer(out.arr), lhs.arr, rhs.arr,
                                       lhs_opts.value, rhs_opts.value))
        return out
