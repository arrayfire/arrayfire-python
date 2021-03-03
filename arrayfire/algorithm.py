#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Vector algorithms (sum, min, sort, etc).
"""

from .library import *
from .array import *

def _parallel_dim(a, dim, c_func):
    out = Array()
    safe_call(c_func(c_pointer(out.arr), a.arr, c_int_t(dim)))
    return out

def _reduce_all(a, c_func):
    real = c_double_t(0)
    imag = c_double_t(0)

    safe_call(c_func(c_pointer(real), c_pointer(imag), a.arr))

    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j

def _nan_parallel_dim(a, dim, c_func, nan_val):
    out = Array()
    safe_call(c_func(c_pointer(out.arr), a.arr, c_int_t(dim), c_double_t(nan_val)))
    return out

def _nan_reduce_all(a, c_func, nan_val):
    real = c_double_t(0)
    imag = c_double_t(0)

    safe_call(c_func(c_pointer(real), c_pointer(imag), a.arr, c_double_t(nan_val)))

    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j

def _FNSD(dim, dims):
    if dim >= 0:
        return int(dim)

    fnsd = 0
    for i, d in enumerate(dims):
        if d > 1:
            fnsd = i
            break
    return int(fnsd)

def _rbk_dim(keys, vals, dim, c_func):
    keys_out = Array()
    vals_out = Array()
    rdim = _FNSD(dim, vals.dims())
    safe_call(c_func(c_pointer(keys_out.arr), c_pointer(vals_out.arr), keys.arr, vals.arr, c_int_t(rdim)))
    return keys_out, vals_out

def _nan_rbk_dim(a, dim, c_func, nan_val):
    keys_out = Array()
    vals_out = Array()
    rdim = _FNSD(dim, vals.dims())
    safe_call(c_func(c_pointer(keys_out.arr), c_pointer(vals_out.arr), keys.arr, vals.arr, c_int_t(rdim), c_double_t(nan_val)))
    return keys_out, vals_out

def sum(a, dim=None, nan_val=None):
    """
    Calculate the sum of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the sum is required.
    nan_val: optional: scalar. default: None
         The value that replaces NaN in the array

    Returns
    -------
    out: af.Array or scalar number
         The sum of all elements in `a` along dimension `dim`.
         If `dim` is `None`, sum of the entire Array is returned.
    """
    if (nan_val is not None):
        if dim is not None:
            return _nan_parallel_dim(a, dim, backend.get().af_sum_nan, nan_val)
        else:
            return _nan_reduce_all(a, backend.get().af_sum_nan_all, nan_val)
    else:
        if dim is not None:
            return _parallel_dim(a, dim, backend.get().af_sum)
        else:
            return _reduce_all(a, backend.get().af_sum_all)


def sumByKey(keys, vals, dim=-1, nan_val=None):
    """
    Calculate the sum of elements along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which the sum will occur.
    nan_val: optional: scalar. default: None
         The value that replaces NaN in the array

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of all elements in `vals` along dimension `dim`.
    values: af.Array or scalar number
         The sum of all elements in `vals` along dimension `dim` according to keys
    """
    if (nan_val is not None):
        return _nan_rbk_dim(keys, vals, dim, backend.get().af_sum_by_key_nan, nan_val)
    else:
        return _rbk_dim(keys, vals, dim, backend.get().af_sum_by_key)

def product(a, dim=None, nan_val=None):
    """
    Calculate the product of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the product is required.
    nan_val: optional: scalar. default: None
         The value that replaces NaN in the array

    Returns
    -------
    out: af.Array or scalar number
         The product of all elements in `a` along dimension `dim`.
         If `dim` is `None`, product of the entire Array is returned.
    """
    if (nan_val is not None):
        if dim is not None:
            return _nan_parallel_dim(a, dim, backend.get().af_product_nan, nan_val)
        else:
            return _nan_reduce_all(a, backend.get().af_product_nan_all, nan_val)
    else:
        if dim is not None:
            return _parallel_dim(a, dim, backend.get().af_product)
        else:
            return _reduce_all(a, backend.get().af_product_all)

def productByKey(keys, vals, dim=-1, nan_val=None):
    """
    Calculate the product of elements along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which the product will occur.
    nan_val: optional: scalar. default: None
         The value that replaces NaN in the array

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of all elements in `vals` along dimension `dim`.
    values: af.Array or scalar number
         The product of all elements in `vals` along dimension `dim` according to keys
    """
    if (nan_val is not None):
        return _nan_rbk_dim(keys, vals, dim, backend.get().af_product_by_key_nan, nan_val)
    else:
        return _rbk_dim(keys, vals, dim, backend.get().af_product_by_key)

def min(a, dim=None):
    """
    Find the minimum value of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the minimum value is required.

    Returns
    -------
    out: af.Array or scalar number
         The minimum value of all elements in `a` along dimension `dim`.
         If `dim` is `None`, minimum value of the entire Array is returned.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_min)
    else:
        return _reduce_all(a, backend.get().af_min_all)

def minByKey(keys, vals, dim=-1):
    """
    Calculate the min of elements along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which the min will occur.

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of all elements in `vals` along dimension `dim`.
    values: af.Array or scalar number
         The min of all elements in `vals` along dimension `dim` according to keys
    """
    return _rbk_dim(keys, vals, dim, backend.get().af_min_by_key)

def max(a, dim=None):
    """
    Find the maximum value of all the elements along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the maximum value is required.

    Returns
    -------
    out: af.Array or scalar number
         The maximum value of all elements in `a` along dimension `dim`.
         If `dim` is `None`, maximum value of the entire Array is returned.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_max)
    else:
        return _reduce_all(a, backend.get().af_max_all)

def maxRagged(vals, lens, dim):
    """
    Find the maximum value of a subset of elements along a specified dimension.

    The size of the subset of elements along the given dimension are decided based on the lengths
    provided in the `lens` array.

    Parameters
    ----------
    vals  : af.Array
         Multi dimensional arrayfire array.
    lens  : af.Array
         Multi dimensional arrayfire array containing number of elements to reduce along given `dim`
    dim: optional: int. default: None
         Dimension along which the maximum value is required.

    Returns
    -------
    (values, indices): A tuple of af.Array(s)
         `values` af.Array will have the maximum values along given dimension for
         subsets determined by lengths provided in `lens`
         `idx` contains the locations of the maximum values as per the lengths provided in `lens`
    """
    out_vals = Array()
    out_idx = Array()
    safe_call(backend().get().af_max_ragged(c_pointer(out_vals.arr), c_pointer(out_idx.arr), c_pointer(vals.arr), c_pointer(lens.arr), c_int_t(dim)))
    return out_vals, out_idx

def maxByKey(keys, vals, dim=-1):
    """
    Calculate the max of elements along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which the max will occur.

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of all elements in `vals` along dimension `dim`.
    values: af.Array or scalar number
         The max of all elements in `vals` along dimension `dim` according to keys.
    """
    return _rbk_dim(keys, vals, dim, backend.get().af_max_by_key)

def all_true(a, dim=None):
    """
    Check if all the elements along a specified dimension are true.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the product is required.

    Returns
    -------
    out: af.Array or scalar number
         Af.array containing True if all elements in `a` along the dimension are True.
         If `dim` is `None`, output is True if `a` does not have any zeros, else False.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_all_true)
    else:
        return _reduce_all(a, backend.get().af_all_true_all)

def allTrueByKey(keys, vals, dim=-1):
    """
    Calculate if all elements are true along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which the all true check will occur.

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of all true check in `vals` along dimension `dim`.
    values: af.Array or scalar number
         Booleans denoting if all elements are true in `vals` along dimension `dim` according to keys
    """
    return _rbk_dim(keys, vals, dim, backend.get().af_all_true_by_key)

def any_true(a, dim=None):
    """
    Check if any the elements along a specified dimension are true.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the product is required.

    Returns
    -------
    out: af.Array or scalar number
         Af.array containing True if any elements in `a` along the dimension are True.
         If `dim` is `None`, output is True if `a` does not have any zeros, else False.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_any_true)
    else:
        return _reduce_all(a, backend.get().af_any_true_all)

def anyTrueByKey(keys, vals, dim=-1):
    """
    Calculate if any elements are true along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which the any true check will occur.

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of any true check in `vals` along dimension `dim`.
    values: af.Array or scalar number
         Booleans denoting if any elements are true in `vals` along dimension `dim` according to keys.
    """
    return _rbk_dim(keys, vals, dim, backend.get().af_any_true_by_key)

def count(a, dim=None):
    """
    Count the number of non zero elements in an array along a specified dimension.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the the non zero elements are to be counted.

    Returns
    -------
    out: af.Array or scalar number
         The count of non zero elements in `a` along `dim`.
         If `dim` is `None`, the total number of non zero elements in `a`.
    """
    if dim is not None:
        return _parallel_dim(a, dim, backend.get().af_count)
    else:
        return _reduce_all(a, backend.get().af_count_all)

def countByKey(keys, vals, dim=-1):
    """
    Counts non-zero elements along a specified dimension according to a key.

    Parameters
    ----------
    keys  : af.Array
         One dimensional arrayfire array with reduction keys.
    vals  : af.Array
         Multi dimensional arrayfire array that will be reduced.
    dim: optional: int. default: -1
         Dimension along which to count elements.

    Returns
    -------
    keys: af.Array or scalar number
         The reduced keys of count in `vals` along dimension `dim`.
    values: af.Array or scalar number
         Count of non-zero elements in `vals` along dimension `dim` according to keys.
    """
    return _rbk_dim(keys, vals, dim, backend.get().af_count_by_key)

def imin(a, dim=None):
    """
    Find the value and location of the minimum value along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the minimum value is required.

    Returns
    -------
    (val, idx): tuple of af.Array or scalars
                `val` contains the minimum value of `a` along `dim`.
                `idx` contains the location of where `val` occurs in `a` along `dim`.
                If `dim` is `None`, `val` and `idx` value and location of global minimum.
    """
    if dim is not None:
        out = Array()
        idx = Array()
        safe_call(backend.get().af_imin(c_pointer(out.arr), c_pointer(idx.arr), a.arr, c_int_t(dim)))
        return out,idx
    else:
        real = c_double_t(0)
        imag = c_double_t(0)
        idx  = c_uint_t(0)
        safe_call(backend.get().af_imin_all(c_pointer(real), c_pointer(imag), c_pointer(idx), a.arr))
        real = real.value
        imag = imag.value
        val = real if imag == 0 else real + imag * 1j
        return val,idx.value

def imax(a, dim=None):
    """
    Find the value and location of the maximum value along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: None
         Dimension along which the maximum value is required.

    Returns
    -------
    (val, idx): tuple of af.Array or scalars
                `val` contains the maximum value of `a` along `dim`.
                `idx` contains the location of where `val` occurs in `a` along `dim`.
                If `dim` is `None`, `val` and `idx` value and location of global maximum.
    """
    if dim is not None:
        out = Array()
        idx = Array()
        safe_call(backend.get().af_imax(c_pointer(out.arr), c_pointer(idx.arr), a.arr, c_int_t(dim)))
        return out,idx
    else:
        real = c_double_t(0)
        imag = c_double_t(0)
        idx  = c_uint_t(0)
        safe_call(backend.get().af_imax_all(c_pointer(real), c_pointer(imag), c_pointer(idx), a.arr))
        real = real.value
        imag = imag.value
        val = real if imag == 0 else real + imag * 1j
        return val,idx.value


def accum(a, dim=0):
    """
    Cumulative sum of an array along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which the cumulative sum is required.

    Returns
    -------
    out: af.Array
         array of same size as `a` containing the cumulative sum along `dim`.
    """
    return _parallel_dim(a, dim, backend.get().af_accum)

def scan(a, dim=0, op=BINARYOP.ADD, inclusive_scan=True):
    """
    Generalized scan of an array.

    Parameters
    ----------
    a   : af.Array
        Multi dimensional arrayfire array.

    dim : optional: int. default: 0
        Dimension along which the scan is performed.

    op  : optional: af.BINARYOP. default: af.BINARYOP.ADD.
        Binary option the scan algorithm uses. Can be one of:
        - af.BINARYOP.ADD
        - af.BINARYOP.MUL
        - af.BINARYOP.MIN
        - af.BINARYOP.MAX

    inclusive_scan: optional: bool. default: True
        Specifies if the scan is inclusive

    Returns
    ---------
    out : af.Array
        - will contain scan of input.
    """
    out = Array()
    safe_call(backend.get().af_scan(c_pointer(out.arr), a.arr, dim, op.value, inclusive_scan))
    return out

def scan_by_key(key, a, dim=0, op=BINARYOP.ADD, inclusive_scan=True):
    """
    Generalized scan by key of an array.

    Parameters
    ----------
    key : af.Array
        key array.

    a   : af.Array
        Multi dimensional arrayfire array.

    dim : optional: int. default: 0
        Dimension along which the scan is performed.

    op  : optional: af.BINARYOP. default: af.BINARYOP.ADD.
        Binary option the scan algorithm uses. Can be one of:
        - af.BINARYOP.ADD
        - af.BINARYOP.MUL
        - af.BINARYOP.MIN
        - af.BINARYOP.MAX

    inclusive_scan: optional: bool. default: True
        Specifies if the scan is inclusive

    Returns
    ---------
    out : af.Array
        - will contain scan of input.
    """
    out = Array()
    safe_call(backend.get().af_scan_by_key(c_pointer(out.arr), key.arr, a.arr, dim, op.value, inclusive_scan))
    return out

def where(a):
    """
    Find the indices of non zero elements

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.

    Returns
    -------
    idx: af.Array
         Linear indices for non zero elements.
    """
    out = Array()
    safe_call(backend.get().af_where(c_pointer(out.arr), a.arr))
    return out

def diff1(a, dim=0):
    """
    Find the first order differences along specified dimensions

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which the differences are required.

    Returns
    -------
    out: af.Array
         Array whose length along `dim` is 1 less than that of `a`.
    """
    return _parallel_dim(a, dim, backend.get().af_diff1)

def diff2(a, dim=0):
    """
    Find the second order differences along specified dimensions

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which the differences are required.

    Returns
    -------
    out: af.Array
         Array whose length along `dim` is 2 less than that of `a`.
    """
    return _parallel_dim(a, dim, backend.get().af_diff2)

def sort(a, dim=0, is_ascending=True):
    """
    Sort the array along a specified dimension

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which sort is to be performed.
    is_ascending: optional: bool. default: True
         Specifies the direction of the sort

    Returns
    -------
    out: af.Array
         array containing the sorted values

    Note
    -------
    Currently `dim` is only supported for 0.
    """
    out = Array()
    safe_call(backend.get().af_sort(c_pointer(out.arr), a.arr, c_uint_t(dim), c_bool_t(is_ascending)))
    return out

def sort_index(a, dim=0, is_ascending=True):
    """
    Sort the array along a specified dimension and get the indices.

    Parameters
    ----------
    a  : af.Array
         Multi dimensional arrayfire array.
    dim: optional: int. default: 0
         Dimension along which sort is to be performed.
    is_ascending: optional: bool. default: True
         Specifies the direction of the sort

    Returns
    -------
    (val, idx): tuple of af.Array
         `val` is an af.Array containing the sorted values.
         `idx` is an af.Array containing the original indices of `val` in `a`.

    Note
    -------
    Currently `dim` is only supported for 0.
    """
    out = Array()
    idx = Array()
    safe_call(backend.get().af_sort_index(c_pointer(out.arr), c_pointer(idx.arr), a.arr,
                                          c_uint_t(dim), c_bool_t(is_ascending)))
    return out,idx

def sort_by_key(ik, iv, dim=0, is_ascending=True):
    """
    Sort an array based on specified keys

    Parameters
    ----------
    ik  : af.Array
         An Array containing the keys
    iv  : af.Array
         An Array containing the values
    dim: optional: int. default: 0
         Dimension along which sort is to be performed.
    is_ascending: optional: bool. default: True
         Specifies the direction of the sort

    Returns
    -------
    (ok, ov): tuple of af.Array
         `ok` contains the values from `ik` in sorted order
         `ov` contains the values from `iv` after sorting them based on `ik`

    Note
    -------
    Currently `dim` is only supported for 0.
    """
    ov = Array()
    ok = Array()
    safe_call(backend.get().af_sort_by_key(c_pointer(ok.arr), c_pointer(ov.arr),
                                           ik.arr, iv.arr, c_uint_t(dim), c_bool_t(is_ascending)))
    return ov,ok

def set_unique(a, is_sorted=False):
    """
    Find the unique elements of an array.

    Parameters
    ----------
    a  : af.Array
         A 1D arrayfire array.
    is_sorted: optional: bool. default: False
         Specifies if the input is pre-sorted.

    Returns
    -------
    out: af.Array
         an array containing the unique values from `a`
    """
    out = Array()
    safe_call(backend.get().af_set_unique(c_pointer(out.arr), a.arr, c_bool_t(is_sorted)))
    return out

def set_union(a, b, is_unique=False):
    """
    Find the union of two arrays.

    Parameters
    ----------
    a  : af.Array
         A 1D arrayfire array.
    b  : af.Array
         A 1D arrayfire array.
    is_unique: optional: bool. default: False
         Specifies if the both inputs contain unique elements.

    Returns
    -------
    out: af.Array
         an array values after performing the union of `a` and `b`.
    """
    out = Array()
    safe_call(backend.get().af_set_union(c_pointer(out.arr), a.arr, b.arr, c_bool_t(is_unique)))
    return out

def set_intersect(a, b, is_unique=False):
    """
    Find the intersect of two arrays.

    Parameters
    ----------
    a  : af.Array
         A 1D arrayfire array.
    b  : af.Array
         A 1D arrayfire array.
    is_unique: optional: bool. default: False
         Specifies if the both inputs contain unique elements.

    Returns
    -------
    out: af.Array
         an array values after performing the intersect of `a` and `b`.
    """
    out = Array()
    safe_call(backend.get().af_set_intersect(c_pointer(out.arr), a.arr, b.arr, c_bool_t(is_unique)))
    return out
