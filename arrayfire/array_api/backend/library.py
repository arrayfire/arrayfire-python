import ctypes
from typing import Any, Type

from ..dtypes import c_dim_t
from . import backend, safe_call

AFArrayPointer = ctypes._Pointer
AFArray = ctypes.c_void_p
Dim_T = Type[c_dim_t]
Pointer = ctypes._Pointer

# Array management


@safe_call
def af_create_handle(arr: AFArrayPointer, ndims: int, dims: Any, type_: int, /) -> Any:  # FIXME dims typing
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga3b8f5cf6fce69aa1574544bc2d44d7d0
    """
    return backend.af_create_handle(arr, ndims, dims, type_)


@safe_call
def af_retain_array(out: AFArrayPointer, in_: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga7ed45b3f881c0f6c80c5cf2af886dbab
    """
    return backend.af_retain_array(out, in_)


@safe_call
def af_create_array(
        arr: AFArrayPointer, data: ctypes.c_void_p, ndims: int, dims: ..., type_: int, /) -> Any:  # FIXME dims typing
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga834be32357616d8ab735087c6f681858
    """
    return backend.af_create_array(arr, data, ndims, dims, type_)


@safe_call
def af_device_array(
        arr: AFArrayPointer, data: ctypes.c_void_p, ndims: int, dims: ..., type_: int, /) -> Any:  # FIXME dims typing
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaad4fc77f872217e7337cb53bfb623cf5
    """
    return backend.af_device_array(arr, data, ndims, dims, type_)


@safe_call
def af_create_strided_array(
        arr: AFArrayPointer, data: ctypes.c_void_p, offset: Dim_T, ndims: int, dims: ..., strides: ...,
        ty: int, location: int, /) -> Any:  # FIXME dims, strides typing
    """
    source: https://arrayfire.org/docs/group__internal__func__create.htm#gad31241a3437b7b8bc3cf49f85e5c4e0c
    """
    return backend.af_create_strided_array(arr, data, offset, ndims, dims, strides, ty, location)


@safe_call
def af_get_type(type_: ..., arr: AFArray, /) -> Any:  # FIXME type typing
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga0dda6898e1c0d9a43efb56cd6a988c9b
    """
    return backend.af_get_type(type_, arr)


@safe_call
def af_get_elements(elems: ..., arr: AFArray, /) -> Any:  # FIXME elems typing
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6845bbe4385a60a606b88f8130252c1f
    """
    return backend.af_get_elements(elems, arr)


@safe_call
def af_get_numdims(result: ..., arr: AFArray, /) -> Any:  # FIXME typing
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefa019d932ff58c2a829ce87edddd2a8
    """
    return backend.af_get_numdims(result, arr)


@safe_call
def af_get_dims(d0: ..., d1: ..., d2: ..., d3: ..., arr: AFArray, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga8b90da50a532837d9763e301b2267348
    """
    return backend.af_get_dims(d0, d1, d2, d3, arr)


@safe_call
def af_get_scalar(output_value: ..., arr: AFArray, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#gaefe2e343a74a84bd43b588218ecc09a3
    """
    return backend.af_get_scalar(output_value, arr)


@safe_call
def af_is_empty(result: ..., arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga19c749e95314e1c77d816ad9952fb680
    """
    return backend.af_is_empty(result, arr)


@safe_call
def af_get_data_ptr(data: ..., arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__c__api__mat.htm#ga6040dc6f0eb127402fbf62c1165f0b9d
    """
    return backend.af_get_data_ptr(data, arr)


# Arrayfire Functions


@safe_call
def af_index_gen(out: AFArrayPointer, in_: AFArray, ndims: int, indices: ..., /) -> Any:  # FIXME indices
    """
    source: https://arrayfire.org/docs/group__index__func__index.htm#ga14a7d149dba0ed0b977335a3df9d91e6
    """
    return backend.af_index_gen(out, in_, ndims, indices)


@safe_call
def af_transpose(out: AFArrayPointer, in_: AFArray, conjugate: bool, /) -> Any:
    """
    https://arrayfire.org/docs/group__blas__func__transpose.htm#ga716b2b9bf190c8f8d0970aef2b57d8e7
    """
    return backend.af_transpose(out, in_, conjugate)


@safe_call
def af_reorder(out: AFArrayPointer, in_: AFArray, x: ..., y: ..., z: ..., w: ..., /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__manip__func__reorder.htm#ga57383f4d00a3a86eab08dddd52c3ad3d
    """
    return backend.af_reorder(out, in_, x, y, z, w)


@safe_call
def af_array_to_string(output: ..., exp: ..., arr: AFArray, precision: int, transpose: bool, /) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__print__func__tostring.htm#ga01f32ef2420b5d4592c6e4b4964b863b
    """
    return backend.af_array_to_string(output, exp, arr, precision, transpose)


@safe_call
def af_free_host(ptr: ...) -> Any:  # FIXME
    """
    source: https://arrayfire.org/docs/group__device__func__free__host.htm#ga3f1149a837a7ebbe8002d5d2244e3370
    """
    return backend.af_free_host(ptr)


@safe_call
def af_constant_complex(arr: AFArrayPointer, real: ..., imag: ..., ndims: int, dims: ..., type_: ..., /) -> Any:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga5a083b1f3cd8a72a41f151de3bdea1a2
    """
    return backend.af_constant_complex(arr, real, imag, ndims, dims, type_)


@safe_call
def af_constant_long(arr: AFArrayPointer, val: ..., ndims: ..., dims: ..., /) -> Any:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga10f1c9fad1ce9e9fefd885d5a1d1fd49
    """
    return backend.af_constant_long(arr, val, ndims, dims)


@safe_call
def af_constant_ulong(arr: AFArrayPointer, val: ..., ndims: ..., dims: ..., /) -> Any:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#ga67af670cc9314589f8134019f5e68809
    """
    return backend.af_constant_ulong(arr, val, ndims, dims)


@safe_call
def af_constant(arr: AFArrayPointer, val: ..., ndims: ..., dims: ..., type_: ..., /) -> Any:
    """
    source: https://arrayfire.org/docs/group__data__func__constant.htm#gafc51b6a98765dd24cd4139f3bde00670
    """
    return backend.af_constant(arr, val, ndims, dims, type_)

# Arithmetic Operators


@safe_call
def af_add(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__add.htm#ga1dfbee755fedd680f4476803ddfe06a7
    """
    return backend.af_add(out, lhs, rhs, batch)


@safe_call
def af_sub(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__sub.htm#ga80ff99a2e186c23614ea9f36ffc6f0a4
    """
    return backend.af_sub(out, lhs, rhs, batch)


@safe_call
def af_mul(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__mul.htm#ga5f7588b2809ff7551d38b6a0bd583a02
    """
    return backend.af_mul(out, lhs, rhs, batch)


@safe_call
def af_div(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__div.htm#ga21f3f97755702692ec8976934e75fde6
    """
    return backend.af_div(out, lhs, rhs, batch)


@safe_call
def af_mod(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__mod.htm#ga01924d1b59d8886e46fabd2dc9b27e0f
    """
    return backend.af_mod(out, lhs, rhs, batch)


@safe_call
def af_pow(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__pow.htm#ga0f28be1a9c8b176a78c4a47f483e7fc6
    """
    return backend.af_pow(out, lhs, rhs, batch)


# Bitwise Operators

@safe_call
def af_bitnot(out: AFArrayPointer, arr: AFArray, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitnot.htm#gaf97e8a38aab59ed2d3a742515467d01e
    """
    return backend.af_bitnot(out, arr)


@safe_call
def af_bitand(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitand.htm#ga45c0779ade4703708596df11cca98800
    """
    return backend.af_bitand(out, lhs, rhs, batch)


@safe_call
def af_bitor(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitor.htm#ga84c99f77d1d83fd53f949b4d67b5b210
    """
    return backend.af_bitor(out, lhs, rhs, batch)


@safe_call
def af_bitxor(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitxor.htm#ga8188620da6b432998e55fdd1fad22100
    """
    return backend.af_bitxor(out, lhs, rhs, batch)


@safe_call
def af_bitshiftl(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftl.htm#ga3139645aafe6f045a5cab454e9c13137
    """
    return backend.af_bitshiftl(out, lhs, rhs, batch)


@safe_call
def af_bitshiftr(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftr.htm#ga4c06b9977ecf96cdfc83b5dfd1ac4895
    """
    return backend.af_bitshiftr(out, lhs, rhs, batch)


@safe_call
def af_lt(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/arith_8h.htm#ae7aa04bf23b32bb11c4bab8bdd637103
    """
    return backend.af_lt(out, lhs, rhs, batch)


@safe_call
def af_le(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__le.htm#gad5535ce64dbed46d0773fd494e84e922
    """
    return backend.af_le(out, lhs, rhs, batch)


@safe_call
def af_gt(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__gt.htm#ga4e65603259515de8939899a163ebaf9e
    """
    return backend.af_gt(out, lhs, rhs, batch)


@safe_call
def af_ge(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__ge.htm#ga4513f212e0b0a22dcf4653e89c85e3d9
    """
    return backend.af_ge(out, lhs, rhs, batch)


@safe_call
def af_eq(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__eq.htm#ga76d2da7716831616bb81effa9e163693
    """
    return backend.af_eq(out, lhs, rhs, batch)


@safe_call
def af_neq(out: AFArrayPointer, lhs: AFArray, rhs: AFArray, batch: bool, /) -> Any:
    """
    source: https://arrayfire.org/docs/group__arith__func__neq.htm#gae4ee8bd06a410f259f1493fb811ce441
    """
    return backend.af_neq(out, lhs, rhs, batch)
