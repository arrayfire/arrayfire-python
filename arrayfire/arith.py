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

def arith_binary_func(lhs, rhs, c_func):
    out = array()

    is_left_array = isinstance(lhs, array)
    is_right_array = isinstance(rhs, array)

    if not (is_left_array or is_right_array):
        TypeError("Atleast one input needs to be of type arrayfire.array")

    elif (is_left_array and is_right_array):
        safe_call(c_func(pointer(out.arr), lhs.arr, rhs.arr, False))

    elif (is_valid_scalar(rhs)):
        ldims = dim4_tuple(lhs.dims())
        lty = lhs.type()
        other = array()
        other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], lty)
        safe_call(c_func(pointer(out.arr), lhs.arr, other.arr, False))

    else:
        rdims = dim4_tuple(rhs.dims())
        rty = rhs.type()
        other = array()
        other.arr = constant_array(lhs, rdims[0], rdims[1], rdims[2], rdims[3], rty)
        safe_call(c_func(pointer(out.arr), lhs.arr, other.arr, False))

    return out

def arith_unary_func(a, c_func):
    out = array()
    safe_call(c_func(pointer(out.arr), a.arr))
    return out

def cast(a, dtype=f32):
    out=array()
    safe_call(clib.af_cast(pointer(out.arr), a.arr, dtype))
    return out

def minof(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_minof)

def maxof(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_maxof)

def rem(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_rem)

def abs(a):
    return arith_unary_func(a, clib.af_abs)

def arg(a):
    return arith_unary_func(a, clib.af_arg)

def sign(a):
    return arith_unary_func(a, clib.af_sign)

def round(a):
    return arith_unary_func(a, clib.af_round)

def trunc(a):
    return arith_unary_func(a, clib.af_trunc)

def floor(a):
    return arith_unary_func(a, clib.af_floor)

def ceil(a):
    return arith_unary_func(a, clib.af_ceil)

def hypot(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_hypot)

def sin(a):
    return arith_unary_func(a, clib.af_sin)

def cos(a):
    return arith_unary_func(a, clib.af_cos)

def tan(a):
    return arith_unary_func(a, clib.af_tan)

def asin(a):
    return arith_unary_func(a, clib.af_asin)

def acos(a):
    return arith_unary_func(a, clib.af_acos)

def atan(a):
    return arith_unary_func(a, clib.af_atan)

def atan2(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_atan2)

def cplx(lhs, rhs=None):
    if rhs is None:
        return arith_unary_func(lhs, clib.af_cplx)
    else:
        return arith_binary_func(lhs, rhs, clib.af_cplx2)

def real(lhs):
    return arith_unary_func(lhs, clib.af_real)

def imag(lhs):
    return arith_unary_func(lhs, clib.af_imag)

def conjg(lhs):
    return arith_unary_func(lhs, clib.af_conjg)

def sinh(a):
    return arith_unary_func(a, clib.af_sinh)

def cosh(a):
    return arith_unary_func(a, clib.af_cosh)

def tanh(a):
    return arith_unary_func(a, clib.af_tanh)

def asinh(a):
    return arith_unary_func(a, clib.af_asinh)

def acosh(a):
    return arith_unary_func(a, clib.af_acosh)

def atanh(a):
    return arith_unary_func(a, clib.af_atanh)

def root(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_root)

def pow(lhs, rhs):
    return arith_binary_func(lhs, rhs, clib.af_pow)

def pow2(a):
    return arith_unary_func(a, clib.af_pow2)

def exp(a):
    return arith_unary_func(a, clib.af_exp)

def expm1(a):
    return arith_unary_func(a, clib.af_expm1)

def erf(a):
    return arith_unary_func(a, clib.af_erf)

def erfc(a):
    return arith_unary_func(a, clib.af_erfc)

def log(a):
    return arith_unary_func(a, clib.af_log)

def log1p(a):
    return arith_unary_func(a, clib.af_log1p)

def log10(a):
    return arith_unary_func(a, clib.af_log10)

def log2(a):
    return arith_unary_func(a, clib.af_log2)

def sqrt(a):
    return arith_unary_func(a, clib.af_sqrt)

def cbrt(a):
    return arith_unary_func(a, clib.af_cbrt)

def factorial(a):
    return arith_unary_func(a, clib.af_factorial)

def tgamma(a):
    return arith_unary_func(a, clib.af_tgamma)

def lgamma(a):
    return arith_unary_func(a, clib.af_lgamma)

def iszero(a):
    return arith_unary_func(a, clib.af_iszero)

def isinf(a):
    return arith_unary_func(a, clib.af_isinf)

def isnan(a):
    return arith_unary_func(a, clib.af_isnan)
