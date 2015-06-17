from .library import *
from .array import *
from .util import *

def randu(d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    clib.af_randu(pointer(out.arr), 4, pointer(dims), dtype)
    return out

def randn(d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    clib.af_randn(pointer(out.arr), 4, pointer(dims), dtype)
    return out

def identity(d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)

    clib.af_identity(pointer(out.arr), 4, pointer(dims), dtype)
    return out

def constant(val, d0, d1=None, d2=None, d3=None, dtype=f32):

    if not isinstance(dtype, c_int):
        raise TypeError("Invalid dtype")

    out = array()
    dims = dim4(d0, d1, d2, d3)
    c_val = c_double(val)
    clib.af_constant(pointer(out.arr), c_val, 4, pointer(dims), dtype)
    return out
