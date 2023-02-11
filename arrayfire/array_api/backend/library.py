import ctypes
from typing import Any, Tuple

from ..device import PointerSource
from ..dtypes import CType, Dtype
from ..dtypes.helpers import CShape, c_dim_t
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


def af_get_dims(d0, d1, d2, d3, arr: AFArray, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga8b90da50a532837d9763e301b2267348
    """
    return backend_api.af_get_dims(d0, d1, d2, d3, arr)


def af_get_scalar(output_value, arr: AFArray, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefe2e343a74a84bd43b588218ecc09a3
    """
    return backend_api.af_get_scalar(output_value, arr)


def af_is_empty(result, arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga19c749e95314e1c77d816ad9952fb680
    """
    return backend_api.af_is_empty(result, arr)


def af_get_data_ptr(data, arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    return backend_api.af_get_data_ptr(data, arr)


# Arrayfire Functions


def af_index_gen(out: AFArrayPointer, in_: AFArray, ndims: int, indices, /) -> Any:  # FIXME indices
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    return backend_api.af_index_gen(out, in_, ndims, indices)


def af_transpose(out: AFArrayPointer, in_: AFArray, conjugate: bool, /) -> Any:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    return backend_api.af_transpose(out, in_, conjugate)


def af_reorder(out: AFArrayPointer, in_: AFArray, x, y, z, w, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__manip__func__reorder.htm#ga57383f4d00a3a86eab08dddd52c3ad3d
    """
    return backend_api.af_reorder(out, in_, x, y, z, w)


def af_array_to_string(output, exp, arr: AFArray, precision: int, transpose: bool, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    """
    return backend_api.af_array_to_string(output, exp, arr, precision, transpose)


def af_free_host(ptr) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__device__func__free__host.htm#ga3f1149a837a7ebbe8002d5d2244e3370
    """
    return backend_api.af_free_host(ptr)
