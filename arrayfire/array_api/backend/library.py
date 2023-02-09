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
