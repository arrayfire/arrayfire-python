from __future__ import annotations

import ctypes
from typing import Tuple, Union

from ..config import is_arch_x86
from . import Dtype
from . import bool as af_bool
from . import complex64, complex128, float32, float64, int64

c_dim_t = ctypes.c_int if is_arch_x86() else ctypes.c_longlong
ShapeType = Tuple[int, ...]


class CShape(tuple):
    def __new__(cls, *args: int) -> CShape:
        cls.original_shape = len(args)
        return tuple.__new__(cls, args)

    def __init__(self, x1: int = 1, x2: int = 1, x3: int = 1, x4: int = 1) -> None:
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.x1, self.x2, self.x3, self.x4}"

    @property
    def c_array(self):  # type: ignore[no-untyped-def]
        c_shape = c_dim_t * 4  # ctypes.c_int | ctypes.c_longlong * 4
        return c_shape(c_dim_t(self.x1), c_dim_t(self.x2), c_dim_t(self.x3), c_dim_t(self.x4))


def to_str(c_str: ctypes.c_char_p) -> str:
    return str(c_str.value.decode("utf-8"))  # type: ignore[union-attr]


def implicit_dtype(number: Union[int, float], array_dtype: Dtype) -> Dtype:
    if isinstance(number, bool):
        number_dtype = af_bool
    if isinstance(number, int):
        number_dtype = int64
    elif isinstance(number, float):
        number_dtype = float64
    elif isinstance(number, complex):
        number_dtype = complex128
    else:
        raise TypeError(f"{type(number)} is not supported and can not be converted to af.Dtype.")

    if not (array_dtype == float32 or array_dtype == complex64):
        return number_dtype

    if number_dtype == float64:
        return float32

    if number_dtype == complex128:
        return complex64

    return number_dtype
