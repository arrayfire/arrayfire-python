import ctypes
from typing import Any, Tuple

from ..device import PointerSource
from ..dtypes import CType, Dtype
from ..dtypes.helpers import CShape, c_dim_t
from . import ArrayBuffer, backend_api, safe_call, safe_call_func

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

    safe_call_func(
        backend_api.af_create_handle(
            ctypes.pointer(out), c_shape.original_shape, ctypes.pointer(c_shape.c_array), dtype.c_api_value)
    )
    return out


def retain_array(arr: AFArray) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga7ed45b3f881c0f6c80c5cf2af886dbab
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_retain_array(ctypes.pointer(out), arr)
    )
    return out


def create_array(shape: Tuple[int, ...], dtype: Dtype, array_buffer: ArrayBuffer, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga834be32357616d8ab735087c6f681858
    """
    out = ctypes.c_void_p(0)
    c_shape = CShape(*shape)

    safe_call_func(
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

    safe_call_func(
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

    safe_call_func(
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

    safe_call_func(
        backend_api.af_get_type(ctypes.pointer(out), arr)
    )
    return out.value


def get_elements(arr: AFArray) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6845bbe4385a60a606b88f8130252c1f
    """
    out = c_dim_t(0)

    safe_call_func(
        backend_api.af_get_elements(ctypes.pointer(out), arr)
    )
    return out.value


def get_numdims(arr: AFArray) -> int:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefa019d932ff58c2a829ce87edddd2a8
    """
    out = ctypes.c_uint(0)

    safe_call_func(
        backend_api.af_get_numdims(ctypes.pointer(out), arr)
    )
    return out.value


@safe_call
def af_get_dims(d0, d1, d2, d3, arr: AFArray, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga8b90da50a532837d9763e301b2267348
    """
    return backend_api.af_get_dims(d0, d1, d2, d3, arr)


@safe_call
def af_get_scalar(output_value, arr: AFArray, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefe2e343a74a84bd43b588218ecc09a3
    """
    return backend_api.af_get_scalar(output_value, arr)


@safe_call
def af_is_empty(result, arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga19c749e95314e1c77d816ad9952fb680
    """
    return backend_api.af_is_empty(result, arr)


@safe_call
def af_get_data_ptr(data, arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    return backend_api.af_get_data_ptr(data, arr)


# Arrayfire Functions


@safe_call
def af_index_gen(out: AFArrayPointer, in_: AFArray, ndims: int, indices, /) -> Any:  # FIXME indices
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    return backend_api.af_index_gen(out, in_, ndims, indices)


@safe_call
def af_transpose(out: AFArrayPointer, in_: AFArray, conjugate: bool, /) -> Any:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    return backend_api.af_transpose(out, in_, conjugate)


@safe_call
def af_reorder(out: AFArrayPointer, in_: AFArray, x, y, z, w, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__manip__func__reorder.htm#ga57383f4d00a3a86eab08dddd52c3ad3d
    """
    return backend_api.af_reorder(out, in_, x, y, z, w)


@safe_call
def af_array_to_string(output, exp, arr: AFArray, precision: int, transpose: bool, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    """
    return backend_api.af_array_to_string(output, exp, arr, precision, transpose)


@safe_call
def af_free_host(ptr) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__device__func__free__host.htm#ga3f1149a837a7ebbe8002d5d2244e3370
    """
    return backend_api.af_free_host(ptr)

# Arithmetic Operators


def add(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__add.htm#ga1dfbee755fedd680f4476803ddfe06a7
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_add(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def sub(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sub.htm#ga80ff99a2e186c23614ea9f36ffc6f0a4
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_sub(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def mul(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mul.htm#ga5f7588b2809ff7551d38b6a0bd583a02
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_mul(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def div(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__div.htm#ga21f3f97755702692ec8976934e75fde6
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_div(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def mod(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mod.htm#ga01924d1b59d8886e46fabd2dc9b27e0f
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_mod(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def pow(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__pow.htm#ga0f28be1a9c8b176a78c4a47f483e7fc6
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_pow(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


# Bitwise Operators

def bitnot(arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitnot.htm#gaf97e8a38aab59ed2d3a742515467d01e
    """
    out = ctypes.c_void_p(0)
    safe_call_func(
        backend_api.af_pow(ctypes.pointer(out), arr)
    )
    return out


def bitand(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitand.htm#ga45c0779ade4703708596df11cca98800
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_bitand(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def bitor(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitor.htm#ga84c99f77d1d83fd53f949b4d67b5b210
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_bitor(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def bitxor(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitxor.htm#ga8188620da6b432998e55fdd1fad22100
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_bitxor(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def bitshiftl(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftl.htm#ga3139645aafe6f045a5cab454e9c13137
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_bitshiftl(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def bitshiftr(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftr.htm#ga4c06b9977ecf96cdfc83b5dfd1ac4895
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_bitshiftr(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def lt(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/arith_8h.htm#ae7aa04bf23b32bb11c4bab8bdd637103
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_lt(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def le(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__le.htm#gad5535ce64dbed46d0773fd494e84e922
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_le(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def gt(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__gt.htm#ga4e65603259515de8939899a163ebaf9e
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_gt(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def ge(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__ge.htm#ga4513f212e0b0a22dcf4653e89c85e3d9
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_ge(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def eq(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__eq.htm#ga76d2da7716831616bb81effa9e163693
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_eq(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out


def neq(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__neq.htm#gae4ee8bd06a410f259f1493fb811ce441
    """
    out = ctypes.c_void_p(0)

    safe_call_func(
        backend_api.af_neq(ctypes.pointer(out), lhs, rhs, _bcast_var)
    )
    return out
