from typing import Callable

from . import backend
from .array_object import Array


class return_copy:
    # TODO merge with process_c_function in array_object
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, x1: Array, x2: Array) -> Array:
        out = Array()
        out.arr = self.func(x1.arr, x2.arr)
        return out


@return_copy
def add(x1: Array, x2: Array, /) -> Array:
    return backend.add(x1, x2)


@return_copy
def sub(x1: Array, x2: Array, /) -> Array:
    return backend.sub(x1, x2)
