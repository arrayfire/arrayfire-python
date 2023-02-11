import ctypes
import enum
from dataclasses import dataclass

from ..dtypes.helpers import c_dim_t, to_str

backend_api = ctypes.CDLL("/opt/arrayfire//lib/libafcpu.3.dylib")  # Mock


def safe_call(c_err: int) -> None:
    if c_err == _ErrorCodes.none.value:
        return

    err_str = ctypes.c_char_p(0)
    backend_api.af_get_last_error(ctypes.pointer(err_str), ctypes.pointer(c_dim_t(0)))
    raise RuntimeError(to_str(err_str))


class _ErrorCodes(enum.Enum):
    none = 0


@dataclass
class ArrayBuffer:
    address: int
    length: int = 0
