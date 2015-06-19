from .library import *
from .array import *

def dim4(d0=1, d1=1, d2=1, d3=1):
    c_dim4 = c_longlong * 4
    out = c_dim4(1, 1, 1, 1)

    for i, dim in enumerate((d0, d1, d2, d3)):
        if (dim is not None): out[i] = dim

    return out

def dim4_tuple(dims):
    assert(isinstance(dims, tuple))
    out = [1]*4

    for i, dim in enumerate(dims):
        out[i] = dim

    return tuple(out)

def print_array(a):
    clib.af_print_array(a.arr)

def is_valid_scalar(a):
    return isinstance(a, float) or isinstance(a, int) or isinstance(a, complex)
