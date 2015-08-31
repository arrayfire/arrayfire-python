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
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                             lhs_opts, rhs_opts))
    return out

def matmulTN(lhs, rhs):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                             AF_MAT_TRANS, AF_MAT_NONE))
    return out

def matmulNT(lhs, rhs):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                             AF_MAT_NONE, AF_MAT_TRANS))
    return out

def matmulTT(lhs, rhs):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                             AF_MAT_TRANS, AF_MAT_TRANS))
    return out

def dot(lhs, rhs, lhs_opts=AF_MAT_NONE, rhs_opts=AF_MAT_NONE):
    out = Array()
    safe_call(backend.get().af_dot(ct.pointer(out.arr), lhs.arr, rhs.arr,
                          lhs_opts, rhs_opts))
    return out
