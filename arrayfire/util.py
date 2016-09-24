#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Utility functions to help with Array metadata.
"""

from .library import *
import numbers

def dim4(d0=1, d1=1, d2=1, d3=1):
    c_dim4 = c_dim_t * 4
    out = c_dim4(1, 1, 1, 1)

    for i, dim in enumerate((d0, d1, d2, d3)):
        if (dim is not None): out[i] = c_dim_t(dim)

    return out

def _is_number(a):
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
    n_value = n_dtype.value

    f64v = Dtype.f64.value
    f32v = Dtype.f32.value
    c32v = Dtype.c32.value
    c64v = Dtype.c64.value

    if n_value == f64v and (a_dtype == f32v or a_dtype == c32v):
        return Dtype.f32

    if n_value == c64v and (a_dtype == f32v or a_dtype == c32v):
        return Dtype.c32

    return n_dtype

def dim4_to_tuple(dims, default=1):
    assert(isinstance(dims, tuple))

    if (default is not None):
        assert(_is_number(default))

    out = [default]*4

    for i, dim in enumerate(dims):
        out[i] = dim

    return tuple(out)

def to_str(c_str):
    return str(c_str.value.decode('utf-8'))

def safe_call(af_error):
    if (af_error != ERR.NONE.value):
        err_str = c_char_ptr_t(0)
        err_len = c_dim_t(0)
        backend.get().af_get_last_error(c_pointer(err_str), c_pointer(err_len))
        raise RuntimeError(to_str(err_str))

def get_version():
    """
    Function to get the version of arrayfire.
    """
    major=c_int_t(0)
    minor=c_int_t(0)
    patch=c_int_t(0)
    safe_call(backend.get().af_get_version(c_pointer(major), c_pointer(minor), c_pointer(patch)))
    return major.value,minor.value,patch.value

def get_reversion():
    """
    Function to get the revision hash of the library.
    """
    return to_str(backend.get().af_get_revision())

to_dtype = {'f' : Dtype.f32,
            'd' : Dtype.f64,
            'b' : Dtype.b8,
            'B' : Dtype.u8,
            'h' : Dtype.s16,
            'H' : Dtype.u16,
            'i' : Dtype.s32,
            'I' : Dtype.u32,
            'l' : Dtype.s64,
            'L' : Dtype.u64,
            'F' : Dtype.c32,
            'D' : Dtype.c64}

to_typecode = {Dtype.f32.value : 'f',
               Dtype.f64.value : 'd',
               Dtype.b8.value : 'b',
               Dtype.u8.value : 'B',
               Dtype.s16.value : 'h',
               Dtype.u16.value : 'H',
               Dtype.s32.value : 'i',
               Dtype.u32.value : 'I',
               Dtype.s64.value : 'l',
               Dtype.u64.value : 'L',
               Dtype.c32.value : 'F',
               Dtype.c64.value : 'D'}

to_c_type = {Dtype.f32.value : c_float_t,
             Dtype.f64.value : c_double_t,
             Dtype.b8.value : c_char_t,
             Dtype.u8.value : c_uchar_t,
             Dtype.s16.value : c_short_t,
             Dtype.u16.value : c_ushort_t,
             Dtype.s32.value : c_int_t,
             Dtype.u32.value : c_uint_t,
             Dtype.s64.value : c_longlong_t,
             Dtype.u64.value : c_ulonglong_t,
             Dtype.c32.value : c_float_t * 2,
             Dtype.c64.value : c_double_t * 2}

to_typename = {Dtype.f32.value : 'float',
               Dtype.f64.value : 'double',
               Dtype.b8.value : 'bool',
               Dtype.u8.value : 'unsigned char',
               Dtype.s16.value : 'short int',
               Dtype.u16.value : 'unsigned short int',
               Dtype.s32.value : 'int',
               Dtype.u32.value : 'unsigned int',
               Dtype.s64.value : 'long int',
               Dtype.u64.value : 'unsigned long int',
               Dtype.c32.value : 'float complex',
               Dtype.c64.value : 'double complex'}
