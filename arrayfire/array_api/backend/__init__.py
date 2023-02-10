import ctypes
from dataclasses import dataclass
from typing import Any, Callable

from ..dtypes.helpers import c_dim_t, to_str
from .constants import ErrorCodes

backend_api = ctypes.CDLL("/opt/arrayfire//lib/libafcpu.3.dylib")  # Mock


class safe_call:
    def __init__(self, c_function: Callable) -> None:
        self.c_function = c_function

    def __call__(self, *args: Any) -> None:
        c_err = self.c_function(*args)
        if c_err == ErrorCodes.none.value:
            return

        err_str = ctypes.c_char_p(0)
        backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
        raise RuntimeError(to_str(err_str))


def safe_call_func(c_err):
    if c_err == ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
    raise RuntimeError(to_str(err_str))


@dataclass
class ArrayBuffer:
    address: int
    length: int = 0
