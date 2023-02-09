import ctypes

from .array_object import Array
from .backend import library


def add(x1: Array, x2: Array, /) -> Array:
    out = Array()
    library.af_add(ctypes.pointer(out.arr), x1.arr, x2.arr, False)
    return out


def sub(x1: Array, x2: Array, /) -> Array:
    out = Array()
    library.af_sub(ctypes.pointer(out.arr), x1.arr, x2.arr, False)
    return out
