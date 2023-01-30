from .array_object import Array
from .dtypes import Dtype

# TODO implement functions


def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    """
    Copies an array to a specified data type irrespective of Type Promotion Rules rules.

    Parameters
    ----------
    x : Array
        Array to cast.
    dtype: Dtype
        Desired data type.
    copy: bool, optional
        Specifies whether to copy an array when the specified dtype matches the data type of the input array x.
        If True, a newly allocated array must always be returned. If False and the specified dtype matches the data
        type of the input array, the input array must be returned; otherwise, a newly allocated array must be returned.
        Default: True.

    Returns
    -------
    out : Array
        An array having the specified data type. The returned array must have the same shape as x.

    Note
    ----
    - Casting floating-point NaN and infinity values to integral data types is not specified and is
    implementation-dependent.
    - Casting a complex floating-point array to a real-valued data type should not be permitted.
    Historically, when casting a complex floating-point array to a real-valued data type, libraries such as NumPy have
    discarded imaginary components such that, for a complex floating-point array x, astype(x) equals astype(real(x))).
    This behavior is considered problematic as the choice to discard the imaginary component is arbitrary and
    introduces more than one way to achieve the same outcome (i.e., for a complex floating-point array x, astype(x) and
    astype(real(x)) versus only astype(imag(x))). Instead, in order to avoid ambiguity and to promote clarity, this
    specification requires that array API consumers explicitly express which component should be cast to a specified
    real-valued data type.
    - When casting a boolean input array to a real-valued data type, a value of True must cast to a real-valued number
    equal to 1, and a value of False must cast to a real-valued number equal to 0.
    When casting a boolean input array to a complex floating-point data type, a value of True must cast to a complex
    number equal to 1 + 0j, and a value of False must cast to a complex number equal to 0 + 0j.
    - When casting a real-valued input array to bool, a value of 0 must cast to False, and a non-zero value must cast
    to True.
    When casting a complex floating-point array to bool, a value of 0 + 0j must cast to False, and all other values
    must cast to True.
    """
    return NotImplemented


def can_cast(from_: Dtype | Array, to: Dtype, /) -> bool:
    """
    Determines if one data type can be cast to another data type according Type Promotion Rules rules.

    Parameters
    ----------
    from_ : Dtype | Array
        Input data type or array from which to cast.
    to : Dtype
        Desired data type.

    Returns
    -------
    out : bool
        True if the cast can occur according to Type Promotion Rules rules; otherwise, False.
    """
    return NotImplemented


def finfo(type: Dtype | Array, /):  # type: ignore[no-untyped-def]
    # TODO add docstring, implementation and return type -> finfo_object
    return NotImplemented


def iinfo(type: Dtype | Array, /):  # type: ignore[no-untyped-def]
    # TODO add docstring, implementation and return type -> iinfo_object
    return NotImplemented


def isdtype(dtype: Dtype, kind: Dtype | str | tuple[Dtype | str, ...]) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type “kind”.

    Parameters
    ----------
    dtype : Dtype
        The input dtype.
    kind : Dtype | str | tuple[Dtype | str, ...]
        Data type kind.
        - If kind is a dtype, the function must return a boolean indicating whether the input dtype is equal to the
        dtype specified by kind.
        - If kind is a string, the function must return a boolean indicating whether the input dtype is of a specified
        data type kind. The following dtype kinds must be supported:
            - bool: boolean data types (e.g., bool).
            - signed integer: signed integer data types (e.g., int8, int16, int32, int64).
            - unsigned integer: unsigned integer data types (e.g., uint8, uint16, uint32, uint64).
            - integral: integer data types. Shorthand for ('signed integer', 'unsigned integer').
            - real floating: real-valued floating-point data types (e.g., float32, float64).
            - complex floating: complex floating-point data types (e.g., complex64, complex128).
            - numeric: numeric data types. Shorthand for ('integral', 'real floating', 'complex floating').
        - If kind is a tuple, the tuple specifies a union of dtypes and/or kinds, and the function must return a
        boolean indicating whether the input dtype is either equal to a specified dtype or belongs to at least one
        specified data type kind.

    Returns
    -------
    out : bool
        Boolean indicating whether a provided dtype is of a specified data type kind.

    Note
    ----
    - A conforming implementation of the array API standard is not limited to only including the dtypes described in
    this specification in the required data type kinds. For example, implementations supporting float16 and bfloat16
    can include float16 and bfloat16 in the real floating data type kind. Similarly, implementations supporting int128
    can include int128 in the signed integer data type kind.
    In short, conforming implementations may extend data type kinds; however, data type kinds must remain consistent
    (e.g., only integer dtypes may belong to integer data type kinds and only floating-point dtypes may belong to
    floating-point data type kinds), and extensions must be clearly documented as such in library documentation.
    """
    return NotImplemented


def result_type(*arrays_and_dtypes: Dtype | Array) -> Dtype:
    """
    Returns the dtype that results from applying the type promotion rules (see Type Promotion Rules) to the arguments.

    Parameters
    ----------
    arrays_and_dtypes: Dtype | Array
        An arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    out : Dtype
        The dtype resulting from an operation involving the input arrays and dtypes.
    """
    return NotImplemented
