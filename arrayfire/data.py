#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from sys import version_info
from .library import *
from .array import *
from .util import *

def constant(val, d0, d1=None, d2=None, d3=None, dtype=f32):
    out = array()
    out.arr = constant_array(val, d0, d1, d2, d3, dtype)
    return out

# Store builtin range function to be used later
brange = range

def range(d0, d1=None, d2=None, d3=None, dim=-1, dtype=f32):

    if not isinstance(dtype, c_int):
        if isinstance(dtype, int):
            dtype = c_int(dtype)
        else:
            raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    safe_call(clib.af_range(pointer(out.arr), 4, pointer(dims), dim, dtype))
    return out


def iota(d0, d1=None, d2=None, d3=None, dim=-1, tile_dims=None, dtype=f32):
    if not isinstance(dtype, c_int):
        if isinstance(dtype, int):
            dtype = c_int(dtype)
        else:
            raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)
    td=[1]*4

    if tile_dims is not None:
        for i in brange(len(tile_dims)):
            td[i] = tile_dims[i]

    tdims = dim4(td[0], td[1], td[2], td[3])

    safe_call(clib.af_iota(pointer(out.arr), 4, pointer(dims), 4, pointer(tdims), dtype))
    return out

def randu(d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        if isinstance(dtype, int):
            dtype = c_int(dtype)
        else:
            raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    safe_call(clib.af_randu(pointer(out.arr), 4, pointer(dims), dtype))
    return out

def randn(d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        if isinstance(dtype, int):
            dtype = c_int(dtype)
        else:
            raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    safe_call(clib.af_randn(pointer(out.arr), 4, pointer(dims), dtype))
    return out

def set_seed(seed=0):
    safe_call(clib.af_set_seed(c_ulonglong(seed)))

def get_seed():
    seed = c_ulonglong(0)
    safe_call(clib.af_get_seed(pointer(seed)))
    return seed.value

def identity(d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        if isinstance(dtype, int):
            dtype = c_int(dtype)
        else:
            raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    safe_call(clib.af_identity(pointer(out.arr), 4, pointer(dims), dtype))
    return out

def diag(a, num=0, extract=True):
    out = array()
    if extract:
        safe_call(clib.af_diag_extract(pointer(out.arr), a.arr, c_int(num)))
    else:
        safe_call(clib.af_diag_create(pointer(out.arr), a.arr, c_int(num)))
    return out

def join(dim, first, second, third=None, fourth=None):
    out = array()
    if (third is None and fourth is None):
        safe_call(clib.af_join(pointer(out.arr), dim, first.arr, second.arr))
    else:
        c_array_vec = dim4(first, second, 0, 0)
        num = 2
        if third is not None:
            c_array_vec[num] = third.arr
            num+=1
        if fourth is not None:
            c_array_vec[num] = fourth.arr
            num+=1

        safe_call(clib.af_join_many(pointer(out.arr), dim, num, pointer(c_array_vec)))


def tile(a, d0, d1=1, d2=1, d3=1):
    out = array()
    safe_call(clib.af_tile(pointer(out.arr), a.arr, d0, d1, d2, d3))
    return out


def reorder(a, d0=1, d1=0, d2=2, d3=3):
    out = array()
    safe_call(clib.af_reorder(pointer(out.arr), a.arr, d0, d1, d2, d3))
    return out

def shift(a, d0, d1=0, d2=0, d3=0):
    out = array()
    safe_call(clib.af_shift(pointer(out.arr), a.arr, d0, d1, d2, d3))
    return out

def moddims(a, d0, d1=1, d2=1, d3=1):
    out = array()
    dims = dim4(d0, d1, d2, d3)
    safe_call(clib.af_moddims(pointer(out.arr), a.arr, 4, pointer(dims)))
    return out

def flat(a):
    out = array()
    safe_call(clib.af_flat(pointer(out.arr), a.arr))
    return out

def flip(a, dim=0):
    out = array()
    safe_call(clib.af_flip(pointer(out.arr), a.arr, c_int(dim)))
    return out

def lower(a, is_unit_diag=False):
    out = array()
    safe_call(clib.af_lower(pointer(out.arr), a.arr, is_unit_diag))
    return out

def upper(a, is_unit_diag=False):
    out = array()
    safe_call(clib.af_upper(pointer(out.arr), a.arr, is_unit_diag))
    return out
