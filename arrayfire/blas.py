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

def matmul(lhs, rhs, lhs_opts=AF_MAT_NONE, rhs_opts=AF_MAT_NONE):
    out = array()
    safe_call(clib.af_matmul(pointer(out.arr), lhs.arr, rhs.arr,\
                             lhs_opts, rhs_opts))
    return out

def matmulTN(lhs, rhs):
    out = array()
    safe_call(clib.af_matmul(pointer(out.arr), lhs.arr, rhs.arr,\
                             AF_MAT_TRANS, AF_MAT_NONE))
    return out

def matmulNT(lhs, rhs):
    out = array()
    safe_call(clib.af_matmul(pointer(out.arr), lhs.arr, rhs.arr,\
                             AF_MAT_NONE, AF_MAT_TRANS))
    return out

def matmulTT(lhs, rhs):
    out = array()
    safe_call(clib.af_matmul(pointer(out.arr), lhs.arr, rhs.arr,\
                             AF_MAT_TRANS, AF_MAT_TRANS))
    return out

def dot(lhs, rhs, lhs_opts=AF_MAT_NONE, rhs_opts=AF_MAT_NONE):
    out = array()
    safe_call(clib.af_dot(pointer(out.arr), lhs.arr, rhs.arr,\
                          lhs_opts, rhs_opts))
    return out

def transpose(a, conj=False):
    out = array()
    safe_call(clib.af_transpose(pointer(out.arr), a.arr, conj))
    return out

def transpose_inplace(a, conj=False):
    safe_call(clib.af_transpose_inplace(a.arr, conj))
