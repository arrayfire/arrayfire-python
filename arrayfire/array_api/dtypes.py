from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Type

from .config import is_arch_x86

c_dim_t = ctypes.c_int if is_arch_x86() else ctypes.c_longlong


@dataclass
class Dtype:
    typecode: str
    c_type: Type[ctypes._SimpleCData]
    typename: str
    c_api_value: int  # Internal use only


# Specification required
# int8 - Not Supported, b8?  # HACK Dtype("i8", ctypes.c_char, "int8", 4)
int16 = Dtype("h", ctypes.c_short, "short int", 10)
int32 = Dtype("i", ctypes.c_int, "int", 5)
int64 = Dtype("l", ctypes.c_longlong, "long int", 8)
uint8 = Dtype("B", ctypes.c_ubyte, "unsigned_char", 7)
uint16 = Dtype("H", ctypes.c_ushort, "unsigned short int", 11)
uint32 = Dtype("I", ctypes.c_uint, "unsigned int", 6)
uint64 = Dtype("L", ctypes.c_ulonglong, "unsigned long int", 9)
float32 = Dtype("f", ctypes.c_float, "float", 0)
float64 = Dtype("d", ctypes.c_double, "double", 2)
complex64 = Dtype("F", ctypes.c_float*2, "float complext", 1)  # type: ignore[arg-type]
complex128 = Dtype("D", ctypes.c_double*2, "double complext", 3)  # type: ignore[arg-type]
bool = Dtype("b", ctypes.c_bool, "bool", 4)

supported_dtypes = [
    int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, complex64, complex128, bool
]


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
