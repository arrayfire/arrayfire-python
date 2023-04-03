import ctypes
from typing import Tuple, Union, cast

from arrayfire.array import _get_indices  # HACK replace with refactored one

from ..device import PointerSource
from ..dtypes import CType, Dtype
from ..dtypes.helpers import CShape, c_dim_t, to_str
from .backend import ArrayBuffer, backend_api, safe_call

AFArrayPointer = ctypes._Pointer
AFArray = ctypes.c_void_p

# HACK, TODO replace for actual bcast_var after refactoring ~ https://github.com/arrayfire/arrayfire/pull/2871
_bcast_var = False

# Array management


def create_handle(shape: Tuple[int, ...], dtype: Dtype, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga3b8f5cf6fce69aa1574544bc2d44d7d0
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_create_handle(
            ctypes.pointer(out), c_shape.original_shape, ctypes.pointer(c_shape.c_array), dtype.c_api_value)
    )
    return out


def retain_array(arr: AFArray) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga7ed45b3f881c0f6c80c5cf2af886dbab
    """
    out = ctypes.c_void_p(0)

    safe_call(
        backend_api.af_retain_array(ctypes.pointer(out), arr)
    )
    return out


def create_array(shape: Tuple[int, ...], dtype: Dtype, array_buffer: ArrayBuffer, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga834be32357616d8ab735087c6f681858
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_create_array(
            ctypes.pointer(out), ctypes.c_void_p(array_buffer.address), c_shape.original_shape,
            ctypes.pointer(c_shape.c_array), dtype.c_api_value)
    )
    return out


def device_array(shape: Tuple[int, ...], dtype: Dtype, array_buffer: ArrayBuffer, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaad4fc77f872217e7337cb53bfb623cf5
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call(
        backend_api.af_device_array(
            ctypes.pointer(out), ctypes.c_void_p(array_buffer.address), c_shape.original_shape,
            ctypes.pointer(c_shape.c_array), dtype.c_api_value)
    )
    return out


def create_strided_array(
        shape: Tuple[int, ...], dtype: Dtype, array_buffer: ArrayBuffer, offset: CType, strides: Tuple[int, ...],
        pointer_source: PointerSource, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__internal__func__create.htm#gad31241a3437b7b8bc3cf49f85e5c4e0c
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    if offset is None:
        offset = c_dim_t(0)

    if strides is None:
        strides = (1, c_shape[0], c_shape[0]*c_shape[1], c_shape[0]*c_shape[1]*c_shape[2])

    if len(strides) < 4:
        strides += (strides[-1], ) * (4 - len(strides))

    safe_call(
        backend_api.af_create_strided_array(
            ctypes.pointer(out), ctypes.c_void_p(array_buffer.address), offset, c_shape.original_shape,
            ctypes.pointer(c_shape.c_array), CShape(*strides).c_array, dtype.c_api_value, pointer_source.value)
    )
    return out


def get_ctype(arr: AFArray) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga0dda6898e1c0d9a43efb56cd6a988c9b
    """
    out = ctypes.c_int()

    safe_call(
        backend_api.af_get_type(ctypes.pointer(out), arr)
    )
    return out.value


def get_elements(arr: AFArray) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6845bbe4385a60a606b88f8130252c1f
    """
    out = c_dim_t(0)

    safe_call(
        backend_api.af_get_elements(ctypes.pointer(out), arr)
    )
    return out.value


def get_numdims(arr: AFArray) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefa019d932ff58c2a829ce87edddd2a8
    """
    out = ctypes.c_uint(0)

    safe_call(
        backend_api.af_get_numdims(ctypes.pointer(out), arr)
    )
    return out.value


def get_dims(arr: AFArray) -> Tuple[int, ...]:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga8b90da50a532837d9763e301b2267348
    """
    d0 = c_dim_t(0)
    d1 = c_dim_t(0)
    d2 = c_dim_t(0)
    d3 = c_dim_t(0)

    safe_call(
        backend_api.af_get_dims(ctypes.pointer(d0), ctypes.pointer(d1), ctypes.pointer(d2), ctypes.pointer(d3), arr)
    )
    return (d0.value, d1.value, d2.value, d3.value)


def get_scalar(arr: AFArray, dtype: Dtype, /) -> Union[None, int, float, bool, complex]:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefe2e343a74a84bd43b588218ecc09a3
    """
    out = dtype.c_type()
    safe_call(
        backend_api.af_get_scalar(ctypes.pointer(out), arr)
    )
    return cast(Union[None, int, float, bool, complex], out.value)


def is_empty(arr: AFArray) -> bool:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga19c749e95314e1c77d816ad9952fb680
    """
    out = ctypes.c_bool()
    safe_call(
        backend_api.af_is_empty(ctypes.pointer(out), arr)
    )
    return out.value


def get_data_ptr(arr: AFArray, size: int, dtype: Dtype, /) -> ctypes.Array:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    c_shape = dtype.c_type * size
    ctypes_array = c_shape()
    safe_call(
        backend_api.af_get_data_ptr(ctypes.pointer(ctypes_array), arr)
    )
    return ctypes_array


# Arrayfire Functions


def index_gen(arr: AFArray, ndims: int, key: Union[int, slice, Tuple[Union[int, slice, ], ...]], /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    out = ctypes.c_void_p(0)
    safe_call(
        backend_api.af_index_gen(ctypes.pointer(out), arr, c_dim_t(ndims), _get_indices(key).pointer)
    )
    return out


def transpose(arr: AFArray, conjugate: bool, /) -> AFArray:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    out = ctypes.c_void_p(0)
    safe_call(
        backend_api.af_transpose(ctypes.pointer(out), arr, conjugate)
    )
    return out


def reorder(arr: AFArray, ndims: int, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__manip__func__reorder.htm#ga57383f4d00a3a86eab08dddd52c3ad3d
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*(tuple(reversed(range(ndims))) + tuple(range(ndims, 4))))
    safe_call(
        backend_api.af_reorder(ctypes.pointer(out), arr, *c_shape)
    )
    return out


def array_as_str(arr: AFArray) -> str:
    """
    source:
    - https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    - https://arrayfire.org/docs/group__device__func__free__host.htm#ga3f1149a837a7ebbe8002d5d2244e3370
    """
    arr_str = ctypes.c_char_p(0)
    safe_call(
        backend_api.af_array_to_string(ctypes.pointer(arr_str), "", arr, 4, True)
    )
    py_str = to_str(arr_str)
    safe_call(
        backend_api.af_free_host(arr_str)
    )
    return py_str
