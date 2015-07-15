#######################################################
# Copyright (c) 2014, ArrayFire
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
