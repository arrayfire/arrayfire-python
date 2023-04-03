from typing import Tuple, Union

from ..array_object import Array
from . import Dtype

# TODO implement functions


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    return NotImplemented


def can_cast(from_: Union[Dtype, Array], to: Dtype, /) -> bool:
    return NotImplemented


def finfo(type: Union[Dtype, Array], /):  # type: ignore[no-untyped-def]
    # NOTE expected return type -> finfo_object
    return NotImplemented


def iinfo(type: Union[Dtype, Array], /):  # type: ignore[no-untyped-def]
    # NOTE expected return type -> iinfo_object
    return NotImplemented


def isdtype(dtype: Dtype, kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]]) -> bool:
    return NotImplemented


def result_type(*arrays_and_dtypes: Union[Dtype, Array]) -> Dtype:
    return NotImplemented
