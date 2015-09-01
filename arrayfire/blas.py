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

def matmul(lhs, rhs, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                                      lhs_opts.value, rhs_opts.value))
    return out

def matmulTN(lhs, rhs):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.TRANS.value, MATPROP.NONE.value))
    return out

def matmulNT(lhs, rhs):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.NONE.value, MATPROP.TRANS.value))
    return out

def matmulTT(lhs, rhs):
    out = Array()
    safe_call(backend.get().af_matmul(ct.pointer(out.arr), lhs.arr, rhs.arr,
                                      MATPROP.TRANS.value, MATPROP.TRANS.value))
    return out

def dot(lhs, rhs, lhs_opts=MATPROP.NONE, rhs_opts=MATPROP.NONE):
    out = Array()
    safe_call(backend.get().af_dot(ct.pointer(out.arr), lhs.arr, rhs.arr,
                                   lhs_opts.value, rhs_opts.value))
    return out
