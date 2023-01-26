import ctypes
import enum
import numbers


class PointerSource(enum.Enum):
    """
    Source of the pointer
    """
    # FIXME
    device = 0
    host = 1


def to_str(c_str: ctypes.c_char_p) -> str:
    return str(c_str.value.decode("utf-8"))  # type: ignore[union-attr]


def is_number(number: int | float | bool | complex) -> bool:
    return isinstance(number, numbers.Number)
