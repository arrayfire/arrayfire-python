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
        return Dtype.b8
    if isinstance(a, int):
        return Dtype.s64
    elif isinstance(a, float):
        return Dtype.f64
    elif isinstance(a, complex):
        return Dtype.c64
    else:
        return to_dtype[a.dtype.char]

def implicit_dtype(number, a_dtype):
    n_dtype = number_dtype(number)
    n_value = Enum_value(n_dtype)

    f64v = Enum_value(Dtype.f64)
    f32v = Enum_value(Dtype.f32)
    c32v = Enum_value(Dtype.c32)
    c64v = Enum_value(Dtype.c64)

    if n_value == f64v and (a_dtype == f32v or a_dtype == c32v):
        return Dtype.f32

    if n_value == c64v and (a_dtype == f32v or a_dtype == c32v):
        return Dtype.c32

    return n_dtype

def dim4_to_tuple(dims, default=1):
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
    if (af_error != Enum_value(ERR.NONE)):
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

typecodes = ('f', 'F', 'd', 'D', 'b', 'B', 'i', 'I', 'l', 'L')
to_dtype = {'f' : Dtype.f32,
            'd' : Dtype.f64,
            'b' : Dtype.b8,
            'B' : Dtype.u8,
            'i' : Dtype.s32,
            'I' : Dtype.u32,
            'l' : Dtype.s64,
            'L' : Dtype.u64,
            'F' : Dtype.c32,
            'D' : Dtype.c64}

to_typecode = {Enum_value(Dtype.f32) : 'f',
               Enum_value(Dtype.f64) : 'd',
               Enum_value(Dtype.b8 ) : 'b',
               Enum_value(Dtype.u8 ) : 'B',
               Enum_value(Dtype.s32) : 'i',
               Enum_value(Dtype.u32) : 'I',
               Enum_value(Dtype.s64) : 'l',
               Enum_value(Dtype.u64) : 'L',
               Enum_value(Dtype.c32) : 'F',
               Enum_value(Dtype.c64) : 'D'}

to_c_type = {Enum_value(Dtype.f32) : ct.c_float,
             Enum_value(Dtype.f64) : ct.c_double,
             Enum_value(Dtype.b8 ) : ct.c_char,
             Enum_value(Dtype.u8 ) : ct.c_ubyte,
             Enum_value(Dtype.s32) : ct.c_int,
             Enum_value(Dtype.u32) : ct.c_uint,
             Enum_value(Dtype.s64) : ct.c_longlong,
             Enum_value(Dtype.u64) : ct.c_ulonglong,
             Enum_value(Dtype.c32) : ct.c_float * 2,
             Enum_value(Dtype.c64) : ct.c_double * 2}
