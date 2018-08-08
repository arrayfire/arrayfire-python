#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Math functions (sin, sqrt, exp, etc).
"""

from .library import *
from .array import *
from .bcast import _bcast_var
from .util import _is_number

def _arith_binary_func(lhs, rhs, c_func):
    out = Array()

    is_left_array = isinstance(lhs, Array)
    is_right_array = isinstance(rhs, Array)

    if not (is_left_array or is_right_array):
        raise TypeError("Atleast one input needs to be of type arrayfire.array")

    elif (is_left_array and is_right_array):
        safe_call(c_func(c_pointer(out.arr), lhs.arr, rhs.arr, _bcast_var.get()))

    elif (_is_number(rhs)):
        ldims = dim4_to_tuple(lhs.dims())
        rty = implicit_dtype(rhs, lhs.type())
        other = Array()
        other.arr = constant_array(rhs, ldims[0], ldims[1], ldims[2], ldims[3], rty)
        safe_call(c_func(c_pointer(out.arr), lhs.arr, other.arr, _bcast_var.get()))

    else:
        rdims = dim4_to_tuple(rhs.dims())
        lty = implicit_dtype(lhs, rhs.type())
        other = Array()
        other.arr = constant_array(lhs, rdims[0], rdims[1], rdims[2], rdims[3], lty)
        safe_call(c_func(c_pointer(out.arr), other.arr, rhs.arr, _bcast_var.get()))

    return out

def _arith_unary_func(a, c_func):
    out = Array()
    safe_call(c_func(c_pointer(out.arr), a.arr))
    return out

def cast(a, dtype):
    """
    Cast an array to a specified type

    Parameters
    ----------
    a    : af.Array
           Multi dimensional arrayfire array.
    dtype: af.Dtype
           Must be one of the following:
               - Dtype.f32 for float
               - Dtype.f64 for double
               - Dtype.b8  for bool
               - Dtype.u8  for unsigned char
               - Dtype.s32 for signed 32 bit integer
               - Dtype.u32 for unsigned 32 bit integer
               - Dtype.s64 for signed 64 bit integer
               - Dtype.u64 for unsigned 64 bit integer
               - Dtype.c32 for 32 bit complex number
               - Dtype.c64 for 64 bit complex number
    Returns
    --------
    out  : af.Array
           array containing the values from `a` after converting to `dtype`.
    """
    out=Array()
    safe_call(backend.get().af_cast(c_pointer(out.arr), a.arr, dtype.value))
    return out

def minof(lhs, rhs):
    """
    Find the minimum value of two inputs at each location.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         array containing the minimum value at each location of the inputs.

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_minof)

def maxof(lhs, rhs):
    """
    Find the maximum value of two inputs at each location.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         array containing the maximum value at each location of the inputs.

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_maxof)

def clamp(val, low, high):
    """
    Clamp the input value between low and high


    Parameters
    ----------
    val  : af.Array
          Multi dimensional arrayfire array to be clamped.

    low  : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number denoting the lower value(s).

    high : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number denoting the higher value(s).
    """
    out = Array()

    is_low_array = isinstance(low, Array)
    is_high_array = isinstance(high, Array)

    vdims = dim4_to_tuple(val.dims())
    vty = val.type()

    if not is_low_array:
        low_arr = constant_array(low, vdims[0], vdims[1], vdims[2], vdims[3], vty)
    else:
        low_arr = low.arr

    if not is_high_array:
        high_arr = constant_array(high, vdims[0], vdims[1], vdims[2], vdims[3], vty)
    else:
        high_arr = high.arr

    safe_call(backend.get().af_clamp(c_pointer(out.arr), val.arr, low_arr, high_arr, _bcast_var.get()))

    return out

def mod(lhs, rhs):
    """
    Find the modulus.
    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.
    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.
    Returns
    --------
    out : af.Array
         Contains the moduli after dividing each value of lhs` with those in `rhs`.
    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_mod)

def rem(lhs, rhs):
    """
    Find the remainder.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         Contains the remainders after dividing each value of lhs` with those in `rhs`.

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_rem)

def abs(a):
    """
    Find the absolute values.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         Contains the absolute values of the inputs.
    """
    return _arith_unary_func(a, backend.get().af_abs)

def arg(a):
    """
    Find the theta value of the inputs in polar co-ordinates.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         Contains the theta values.
    """
    return _arith_unary_func(a, backend.get().af_arg)

def sign(a):
    """
    Find the sign of the inputs.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing 1 for negative values, 0 otherwise.
    """
    return _arith_unary_func(a, backend.get().af_sign)

def round(a):
    """
    Round the values to nearest integer.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the values rounded to nearest integer.
    """
    return _arith_unary_func(a, backend.get().af_round)

def trunc(a):
    """
    Round the values towards zero.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the truncated values.
    """
    return _arith_unary_func(a, backend.get().af_trunc)

def floor(a):
    """
    Round the values towards a smaller integer.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the floored values.
    """
    return _arith_unary_func(a, backend.get().af_floor)

def ceil(a):
    """
    Round the values towards a bigger integer.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the ceiled values.
    """
    return _arith_unary_func(a, backend.get().af_ceil)

def hypot(lhs, rhs):
    """
    Find the value of the hypotunese.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         Contains the value of `sqrt(lhs**2, rhs**2)`.

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_hypot)

def sin(a):
    """
    Sine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the sine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_sin)

def cos(a):
    """
    Cosine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the cosine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_cos)

def tan(a):
    """
    Tangent of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the tangent of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_tan)

def asin(a):
    """
    Arc Sine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the arc sine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_asin)

def acos(a):
    """
    Arc Cosine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the arc cosine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_acos)

def atan(a):
    """
    Arc Tangent of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the arc tangent of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_atan)

def atan2(lhs, rhs):
    """
    Find the arc tan using two values.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         Contains the value arc tan values where:
         - `lhs` contains the sine values.
         - `rhs` contains the cosine values.

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_atan2)

def cplx(lhs, rhs=None):
    """
    Create a complex array from real inputs.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : optional: af.Array or scalar. default: None.
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         Contains complex values whose
         - real values contain values from `lhs`
         - imaginary values contain values from `rhs` (0 if `rhs` is None)

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    if rhs is None:
        return _arith_unary_func(lhs, backend.get().af_cplx)
    else:
        return _arith_binary_func(lhs, rhs, backend.get().af_cplx2)

def real(a):
    """
    Find the real values of the input.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the real values from `a`.

    """
    return _arith_unary_func(a, backend.get().af_real)

def imag(a):
    """
    Find the imaginary values of the input.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the imaginary values from `a`.
    """
    return _arith_unary_func(a, backend.get().af_imag)

def conjg(a):
    """
    Find the complex conjugate values of the input.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing copmplex conjugate values from `a`.
    """
    return _arith_unary_func(a, backend.get().af_conjg)

def sinh(a):
    """
    Hyperbolic Sine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the hyperbolic sine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_sinh)

def cosh(a):
    """
    Hyperbolic Cosine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the hyperbolic cosine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_cosh)

def tanh(a):
    """
    Hyperbolic Tangent of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the hyperbolic tangent of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_tanh)

def asinh(a):
    """
    Arc Hyperbolic Sine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the arc hyperbolic sine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_asinh)

def acosh(a):
    """
    Arc Hyperbolic Cosine of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the arc hyperbolic cosine of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_acosh)

def atanh(a):
    """
    Arc Hyperbolic Tangent of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the arc hyperbolic tangent of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_atanh)

def root(lhs, rhs):
    """
    Find the root values of two inputs at each location.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         array containing the value of `lhs ** (1/rhs)`

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_root)

def pow(lhs, rhs):
    """
    Find the power of two inputs at each location.

    Parameters
    ----------
    lhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    rhs : af.Array or scalar
          Multi dimensional arrayfire array or a scalar number.

    Returns
    --------
    out : af.Array
         array containing the value of `lhs ** (rhs)`

    Note
    -------
    - Atleast one of `lhs` and `rhs` needs to be af.Array.
    - If `lhs` and `rhs` are both af.Array, they must be of same size.
    """
    return _arith_binary_func(lhs, rhs, backend.get().af_pow)

def pow2(a):
    """
    Raise 2 to the power of each element in input.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array where each element is 2 raised to power of the corresponding value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_pow2)

def sigmoid(a):
    """
    Raise 2 to the power of each element in input.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array where each element is outout of a sigmoid function for the corresponding value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_sigmoid)

def exp(a):
    """
    Exponential of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the exponential of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_exp)

def expm1(a):
    """
    Exponential of each element in the array minus 1.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the exponential of each value from `a`.

    Note
    -------
    - `a` must not be complex.
    - This function provides a more stable result for small values of `a`.
    """
    return _arith_unary_func(a, backend.get().af_expm1)

def erf(a):
    """
    Error function of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the error function of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_erf)

def erfc(a):
    """
    Complementary error function of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the complementary error function of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_erfc)

def log(a):
    """
    Natural logarithm of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the natural logarithm of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_log)

def log1p(a):
    """
    Logarithm of each element in the array plus 1.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the the values of `log(a) + 1`

    Note
    -------
    - `a` must not be complex.
    - This function provides a more stable result for small values of `a`.
    """
    return _arith_unary_func(a, backend.get().af_log1p)

def log10(a):
    """
    Logarithm base 10 of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the logarithm base 10 of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_log10)

def log2(a):
    """
    Logarithm base 2 of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the logarithm base 2 of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_log2)

def sqrt(a):
    """
    Square root of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the square root of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_sqrt)

def cbrt(a):
    """
    Cube root of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the cube root of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_cbrt)

def factorial(a):
    """
    factorial of each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the factorial of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_factorial)

def tgamma(a):
    """
    Performs the gamma function for each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the output of gamma function of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_tgamma)

def lgamma(a):
    """
    Performs the logarithm of gamma function for each element in the array.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the output of logarithm of gamma function of each value from `a`.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_lgamma)

def iszero(a):
    """
    Check if each element of the input is zero.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the output after checking if each value of `a` is 0.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_iszero)

def isinf(a):
    """
    Check if each element of the input is infinity.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the output after checking if each value of `a` is inifnite.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_isinf)

def isnan(a):
    """
    Check if each element of the input is NaN.

    Parameters
    ----------
    a : af.Array
        Multi dimensional arrayfire array.

    Returns
    --------
    out : af.Array
         array containing the output after checking if each value of `a` is NaN.

    Note
    -------
    `a` must not be complex.
    """
    return _arith_unary_func(a, backend.get().af_isnan)
