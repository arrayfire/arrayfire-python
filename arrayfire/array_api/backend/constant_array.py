import ctypes
from typing import Tuple, Union

from ..dtypes import Dtype, int64, uint64
from ..dtypes.helpers import CShape, implicit_dtype
from .backend import backend_api, safe_call

AFArray = ctypes.c_void_p


def _constant_complex(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga5a083b1f3cd8a72a41f151de3bdea1a2
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_constant_complex(
            ctypes.pointer(out), ctypes.c_double(number.real), ctypes.c_double(number.imag), 4,
            ctypes.pointer(c_shape.c_array), dtype.c_api_value)
    )
    return out


def _constant_long(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga10f1c9fad1ce9e9fefd885d5a1d1fd49
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_constant_long(
            ctypes.pointer(out), ctypes.c_longlong(number.real), 4, ctypes.pointer(c_shape.c_array))
    )
    return out


def _constant_ulong(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga67af670cc9314589f8134019f5e68809
    """
    # return backend_api.af_constant_ulong(arr, val, ndims, dims)
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_constant_ulong(
            ctypes.pointer(out), ctypes.c_ulonglong(number.real), 4, ctypes.pointer(c_shape.c_array))
    )
    return out


def _constant(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#gafc51b6a98765dd24cd4139f3bde00670
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_constant(
            ctypes.pointer(out), ctypes.c_double(number), 4, ctypes.pointer(c_shape.c_array), dtype.c_api_value)
    )
    return out


def create_constant_array(number: Union[int, float], shape: Tuple[int, ...], dtype: Dtype, /) -> AFArray:
    dtype = implicit_dtype(number, dtype)

    # NOTE complex is not supported in Data API
    # if isinstance(number, complex):
    #     if dtype != complex64 and dtype != complex128:
    #         dtype = complex64
    #     return _constant_complex(number, shape, dtype)

    if dtype == int64:
        return _constant_long(number, shape, dtype)

    if dtype == uint64:
        return _constant_ulong(number, shape, dtype)

    return _constant(number, shape, dtype)
