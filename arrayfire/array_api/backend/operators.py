import ctypes
from typing import Callable

from .backend import backend_api, safe_call

AFArray = ctypes.c_void_p

# HACK, TODO replace for actual bcast_var after refactoring ~ https://github.com/arrayfire/arrayfire/pull/2871
_bcast_var = False

# Arithmetic Operators


def add(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__add.htm#ga1dfbee755fedd680f4476803ddfe06a7
    """
    return _binary_op(backend_api.af_add, lhs, rhs)


def sub(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__sub.htm#ga80ff99a2e186c23614ea9f36ffc6f0a4
    """
    return _binary_op(backend_api.af_sub, lhs, rhs)


def mul(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mul.htm#ga5f7588b2809ff7551d38b6a0bd583a02
    """
    return _binary_op(backend_api.af_mul, lhs, rhs)


def div(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__div.htm#ga21f3f97755702692ec8976934e75fde6
    """
    return _binary_op(backend_api.af_div, lhs, rhs)


def mod(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__mod.htm#ga01924d1b59d8886e46fabd2dc9b27e0f
    """
    return _binary_op(backend_api.af_mod, lhs, rhs)


def pow(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__pow.htm#ga0f28be1a9c8b176a78c4a47f483e7fc6
    """
    return _binary_op(backend_api.af_pow, lhs, rhs)


# Bitwise Operators

def bitnot(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitnot.htm#gaf97e8a38aab59ed2d3a742515467d01e
    """
    out = ctypes.c_void_p(0)
    safe_call(
        backend_api.af_bitnot(ctypes.pointer(out), arr)
    )
    return out


def bitand(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitand.htm#ga45c0779ade4703708596df11cca98800
    """
    return _binary_op(backend_api.af_bitand, lhs, rhs)


def bitor(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitor.htm#ga84c99f77d1d83fd53f949b4d67b5b210
    """
    return _binary_op(backend_api.af_bitor, lhs, rhs)


def bitxor(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__bitxor.htm#ga8188620da6b432998e55fdd1fad22100
    """
    return _binary_op(backend_api.af_bitxor, lhs, rhs)


def bitshiftl(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftl.htm#ga3139645aafe6f045a5cab454e9c13137
    """
    return _binary_op(backend_api.af_butshiftl, lhs, rhs)


def bitshiftr(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__shiftr.htm#ga4c06b9977ecf96cdfc83b5dfd1ac4895
    """
    return _binary_op(backend_api.af_bitshiftr, lhs, rhs)


def lt(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/arith_8h.htm#ae7aa04bf23b32bb11c4bab8bdd637103
    """
    return _binary_op(backend_api.af_lt, lhs, rhs)


def le(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__le.htm#gad5535ce64dbed46d0773fd494e84e922
    """
    return _binary_op(backend_api.af_le, lhs, rhs)


def gt(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__gt.htm#ga4e65603259515de8939899a163ebaf9e
    """
    return _binary_op(backend_api.af_gt, lhs, rhs)


def ge(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__ge.htm#ga4513f212e0b0a22dcf4653e89c85e3d9
    """
    return _binary_op(backend_api.af_ge, lhs, rhs)


def eq(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__eq.htm#ga76d2da7716831616bb81effa9e163693
    """
    return _binary_op(backend_api.af_eq, lhs, rhs)


def neq(lhs: AFArray, rhs: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__arith__func__neq.htm#gae4ee8bd06a410f259f1493fb811ce441
    """
    return _binary_op(backend_api.af_neq, lhs, rhs)


def _binary_op(c_func: Callable, lhs: AFArray, rhs: AFArray, /) -> AFArray:
    out = ctypes.c_void_p(0)
    safe_call(c_func(ctypes.pointer(out), lhs, rhs, _bcast_var))
    return out
