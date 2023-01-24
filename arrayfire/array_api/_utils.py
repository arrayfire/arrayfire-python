import ctypes
import enum
import numbers
from typing import Any


class Device(enum.Enum):
    # HACK. TODO make it real
    cpu = "cpu"
    gpu = "gpu"


class PointerSource(enum.Enum):
    """
    Source of the pointer
    """
    # FIXME
    device = 0
    host = 1


def to_str(c_str: ctypes.c_char_p) -> str:
    return str(c_str.value.decode("utf-8"))  # type: ignore[union-attr]


def is_number(number: Any) -> bool:
    return isinstance(number, numbers.Number)
