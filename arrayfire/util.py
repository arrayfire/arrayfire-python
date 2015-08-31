#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from .library import *
import numbers

def dim4(d0=1, d1=1, d2=1, d3=1):
    c_dim4 = ct.c_longlong * 4
    out = c_dim4(1, 1, 1, 1)

    for i, dim in enumerate((d0, d1, d2, d3)):
        if (dim is not None): out[i] = dim

    return out

def is_number(a):
    return isinstance(a, numbers.Number)

def number_dtype(a):
    if isinstance(a, bool):
        return b8
    if isinstance(a, int):
        return s64
    elif isinstance(a, float):
        return f64
    elif isinstance(a, complex):
        return c64
    else:
        return to_dtype[a.dtype.char]

def implicit_dtype(number, a_dtype):
    n_dtype = number_dtype(number)
    n_value = n_dtype.value

    f64v = f64.value
    f32v = f32.value
    c32v = c32.value
    c64v = c64.value

    if n_value == f64v and (a_dtype == f32v or a_dtype == c32v):
        return f32

    if n_value == c64v and (a_dtype == f32v or a_dtype == c32v):
        return c32

    return n_dtype

def dim4_tuple(dims, default=1):
    assert(isinstance(dims, tuple))

    if (default is not None):
        assert(is_number(default))

    out = [default]*4

    for i, dim in enumerate(dims):
        out[i] = dim

    return tuple(out)

def to_str(c_str):
    return str(c_str.value.decode('utf-8'))

def safe_call(af_error):
    if (af_error != AF_SUCCESS.value):
        err_str = ct.c_char_p(0)
        err_len = ct.c_longlong(0)
        backend.get().af_get_last_error(ct.pointer(err_str), ct.pointer(err_len))
        raise RuntimeError(to_str(err_str), af_error)

def get_version():
    major=ct.c_int(0)
    minor=ct.c_int(0)
    patch=ct.c_int(0)
    safe_call(backend.get().af_get_version(ct.pointer(major), ct.pointer(minor), ct.pointer(patch)))
    return major,minor,patch

to_dtype = {'f' : f32,
            'd' : f64,
            'b' : b8,
            'B' : u8,
            'i' : s32,
            'I' : u32,
            'l' : s64,
            'L' : u64,
            'F' : c32,
            'D' : c64}

to_typecode = {f32.value : 'f',
               f64.value : 'd',
               b8.value : 'b',
               u8.value : 'B',
               s32.value : 'i',
               u32.value : 'I',
               s64.value : 'l',
               u64.value : 'L',
               c32.value : 'F',
               c64.value : 'D'}

to_c_type = {f32.value : ct.c_float,
             f64.value : ct.c_double,
             b8.value : ct.c_char,
             u8.value : ct.c_ubyte,
             s32.value : ct.c_int,
             u32.value : ct.c_uint,
             s64.value : ct.c_longlong,
             u64.value : ct.c_ulonglong,
             c32.value : ct.c_float * 2,
             c64.value : ct.c_double * 2}
