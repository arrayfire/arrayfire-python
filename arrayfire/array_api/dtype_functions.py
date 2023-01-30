from .array_object import Array
from .dtypes import Dtype

# TODO implement functions


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    return NotImplemented


def can_cast(from_: Dtype | Array, to: Dtype, /) -> bool:
    return NotImplemented


def finfo(type: Dtype | Array, /):  # type: ignore[no-untyped-def]
    # NOTE expected return type -> finfo_object
    return NotImplemented


def iinfo(type: Dtype | Array, /):  # type: ignore[no-untyped-def]
    # NOTE expected return type -> iinfo_object
    return NotImplemented


def isdtype(dtype: Dtype, kind: Dtype | str | tuple[Dtype | str, ...]) -> bool:
    return NotImplemented


def result_type(*arrays_and_dtypes: Dtype | Array) -> Dtype:
    return NotImplemented
