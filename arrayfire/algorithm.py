#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from .library import *
from .array import *

def parallel_dim(a, dim, c_func):
    out = Array()
    safe_call(c_func(ct.pointer(out.arr), a.arr, ct.c_int(dim)))
    return out

def reduce_all(a, c_func):
    real = ct.c_double(0)
    imag = ct.c_double(0)
    safe_call(c_func(ct.pointer(real), ct.pointer(imag), a.arr))
    real = real.value
    imag = imag.value
    return real if imag == 0 else real + imag * 1j

def sum(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_sum)
    else:
        return reduce_all(a, clib.af_sum_all)

def product(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_product)
    else:
        return reduce_all(a, clib.af_product_all)

def min(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_min)
    else:
        return reduce_all(a, clib.af_min_all)

def max(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_max)
    else:
        return reduce_all(a, clib.af_max_all)

def all_true(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_all_true)
    else:
        return reduce_all(a, clib.af_all_true_all)

def any_true(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_any_true)
    else:
        return reduce_all(a, clib.af_any_true_all)

def count(a, dim=None):
    if dim is not None:
        return parallel_dim(a, dim, clib.af_count)
    else:
        return reduce_all(a, clib.af_count_all)

def imin(a, dim=None):
    if dim is not None:
        out = Array()
        idx = Array()
        safe_call(clib.af_imin(ct.pointer(out.arr), ct.pointer(idx.arr), a.arr, ct.c_int(dim)))
        return out,idx
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)
        idx  = ct.c_uint(0)
        safe_call(clib.af_imin_all(ct.pointer(real), ct.pointer(imag), ct.pointer(idx), a.arr))
        real = real.value
        imag = imag.value
        val = real if imag == 0 else real + imag * 1j
        return val,idx.value

def imax(a, dim=None):
    if dim is not None:
        out = Array()
        idx = Array()
        safe_call(clib.af_imax(ct.pointer(out.arr), ct.pointer(idx.arr), a.arr, ct.c_int(dim)))
        return out,idx
    else:
        real = ct.c_double(0)
        imag = ct.c_double(0)
        idx  = ct.c_uint(0)
        safe_call(clib.af_imax_all(ct.pointer(real), ct.pointer(imag), ct.pointer(idx), a.arr))
        real = real.value
        imag = imag.value
        val = real if imag == 0 else real + imag * 1j
        return val,idx.value


def accum(a, dim=0):
    return parallel_dim(a, dim, clib.af_accum)

def where(a):
    out = Array()
    safe_call(clib.af_where(ct.pointer(out.arr), a.arr))
    return out

def diff1(a, dim=0):
    return parallel_dim(a, dim, clib.af_diff1)

def diff2(a, dim=0):
    return parallel_dim(a, dim, clib.af_diff2)

def sort(a, dim=0, is_ascending=True):
    out = Array()
    safe_call(clib.af_sort(ct.pointer(out.arr), a.arr, ct.c_uint(dim), ct.c_bool(is_ascending)))
    return out

def sort_index(a, dim=0, is_ascending=True):
    out = Array()
    idx = Array()
    safe_call(clib.af_sort_index(ct.pointer(out.arr), ct.pointer(idx.arr), a.arr,
                                 ct.c_uint(dim), ct.c_bool(is_ascending)))
    return out,idx

def sort_by_key(iv, ik, dim=0, is_ascending=True):
    ov = Array()
    ok = Array()
    safe_call(clib.af_sort_by_key(ct.pointer(ov.arr), ct.pointer(ok.arr),
                                  iv.arr, ik.arr, ct.c_uint(dim), ct.c_bool(is_ascending)))
    return ov,ok

def set_unique(a, is_sorted=False):
    out = Array()
    safe_call(clib.af_set_unique(ct.pointer(out.arr), a.arr, ct.c_bool(is_sorted)))
    return out

def set_union(a, b, is_unique=False):
    out = Array()
    safe_call(clib.af_set_union(ct.pointer(out.arr), a.arr, b.arr, ct.c_bool(is_unique)))
    return out

def set_intersect(a, b, is_unique=False):
    out = Array()
    safe_call(clib.af_set_intersect(ct.pointer(out.arr), a.arr, b.arr, ct.c_bool(is_unique)))
    return out
