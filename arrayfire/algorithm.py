from .library import *
from .array import *

def sum(A, dim=0):
    out = array()
    clib.af_sum(pointer(out.arr), A.arr, c_int(dim))
    return out

def min(A, dim=0):
    out = array()
    clib.af_min(pointer(out.arr), A.arr, c_int(dim))
    return out

def max(A, dim=0):
    out = array()
    clib.af_max(pointer(out.arr), A.arr, c_int(dim))
    return out

def any_true(A, dim=0):
    out = array()
    clib.af_any_true(pointer(out.arr), A.arr, c_int(dim))
    return out

def all_true(A, dim=0):
    out = array()
    clib.af_all_true(pointer(out.arr), A.arr, c_int(dim))
    return out

def accum(A, dim=0):
    out = array()
    clib.af_accum(pointer(out.arr), A.arr, c_int(dim))
    return out

def sort(A, dim=0):
    out = array()
    clib.af_sort(pointer(out.arr), A.arr, c_int(dim))
    return out

def diff1(A, dim=0):
    out = array()
    clib.af_diff1(pointer(out.arr), A.arr, c_int(dim))
    return out

def diff2(A, dim=0):
    out = array()
    clib.af_diff2(pointer(out.arr), A.arr, c_int(dim))
    return out

def where(A):
    out = array()
    clib.af_where(pointer(out.arr), A.arr)
    return out
