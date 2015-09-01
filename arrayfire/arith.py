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
from .broadcast import *

def _arith_binary_func(lhs, rhs, c_func):
    out = Array()

    is_left_array = isinstance(lhs, Array)
    is_right_array = isinstance(rhs, Array)

    if not (is_left_array or is_right_array):
        raise TypeError("Atleast one input needs to be of type arrayfire.array")

    elif (is_left_array and is_right_array):
        safe_call(c_func(ct.pointer(out.arr), lhs.arr, rhs.arr, bcast.get()))

    elif (is_number(rhs)):
        ldims = dim4_to_tuple(lhs.dims())
        rty = implicit_dtype(rhs, lhs.type())
        other = Array()
        other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], rty)
        safe_call(c_func(ct.pointer(out.arr), lhs.arr, other.arr, bcast.get()))

    else:
        rdims = dim4_to_tuple(rhs.dims())
        lty = implicit_dtype(lhs, rhs.type())
        other = Array()
        other.arr = constant_array(lhs, rdims[0], rdims[1], rdims[2], rdims[3], lty)
        safe_call(c_func(ct.pointer(out.arr), other.arr, rhs.arr, bcast.get()))

    return out

def _arith_unary_func(a, c_func):
    out = Array()
    safe_call(c_func(ct.pointer(out.arr), a.arr))
    return out

def cast(a, dtype=f32):
    out=Array()
    safe_call(backend.get().af_cast(ct.pointer(out.arr), a.arr, dtype))
    return out

def minof(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_minof)

def maxof(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_maxof)

def rem(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_rem)

def abs(a):
    return _arith_unary_func(a, backend.get().af_abs)

def arg(a):
    return _arith_unary_func(a, backend.get().af_arg)

def sign(a):
    return _arith_unary_func(a, backend.get().af_sign)

def round(a):
    return _arith_unary_func(a, backend.get().af_round)

def trunc(a):
    return _arith_unary_func(a, backend.get().af_trunc)

def floor(a):
    return _arith_unary_func(a, backend.get().af_floor)

def ceil(a):
    return _arith_unary_func(a, backend.get().af_ceil)

def hypot(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_hypot)

def sin(a):
    return _arith_unary_func(a, backend.get().af_sin)

def cos(a):
    return _arith_unary_func(a, backend.get().af_cos)

def tan(a):
    return _arith_unary_func(a, backend.get().af_tan)

def asin(a):
    return _arith_unary_func(a, backend.get().af_asin)

def acos(a):
    return _arith_unary_func(a, backend.get().af_acos)

def atan(a):
    return _arith_unary_func(a, backend.get().af_atan)

def atan2(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_atan2)

def cplx(lhs, rhs=None):
    if rhs is None:
        return _arith_unary_func(lhs, backend.get().af_cplx)
    else:
        return _arith_binary_func(lhs, rhs, backend.get().af_cplx2)

def real(lhs):
    return _arith_unary_func(lhs, backend.get().af_real)

def imag(lhs):
    return _arith_unary_func(lhs, backend.get().af_imag)

def conjg(lhs):
    return _arith_unary_func(lhs, backend.get().af_conjg)

def sinh(a):
    return _arith_unary_func(a, backend.get().af_sinh)

def cosh(a):
    return _arith_unary_func(a, backend.get().af_cosh)

def tanh(a):
    return _arith_unary_func(a, backend.get().af_tanh)

def asinh(a):
    return _arith_unary_func(a, backend.get().af_asinh)

def acosh(a):
    return _arith_unary_func(a, backend.get().af_acosh)

def atanh(a):
    return _arith_unary_func(a, backend.get().af_atanh)

def root(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_root)

def pow(lhs, rhs):
    return _arith_binary_func(lhs, rhs, backend.get().af_pow)

def pow2(a):
    return _arith_unary_func(a, backend.get().af_pow2)

def exp(a):
    return _arith_unary_func(a, backend.get().af_exp)

def expm1(a):
    return _arith_unary_func(a, backend.get().af_expm1)

def erf(a):
    return _arith_unary_func(a, backend.get().af_erf)

def erfc(a):
    return _arith_unary_func(a, backend.get().af_erfc)

def log(a):
    return _arith_unary_func(a, backend.get().af_log)

def log1p(a):
    return _arith_unary_func(a, backend.get().af_log1p)

def log10(a):
    return _arith_unary_func(a, backend.get().af_log10)

def log2(a):
    return _arith_unary_func(a, backend.get().af_log2)

def sqrt(a):
    return _arith_unary_func(a, backend.get().af_sqrt)

def cbrt(a):
    return _arith_unary_func(a, backend.get().af_cbrt)

def factorial(a):
    return _arith_unary_func(a, backend.get().af_factorial)

def tgamma(a):
    return _arith_unary_func(a, backend.get().af_tgamma)

def lgamma(a):
    return _arith_unary_func(a, backend.get().af_lgamma)

def iszero(a):
    return _arith_unary_func(a, backend.get().af_iszero)

def isinf(a):
    return _arith_unary_func(a, backend.get().af_isinf)

def isnan(a):
    return _arith_unary_func(a, backend.get().af_isnan)
