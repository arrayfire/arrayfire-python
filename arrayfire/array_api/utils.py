from .array_object import Array

# TODO implement functions


def all(x: Array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False) -> Array:
    """
    Tests whether all input array elements evaluate to True along a specified axis.

    Parameters
    ----------
    x : Array
        Input array.
    axis :  None | int | tuple[int, ...], optional
        Axis or axes along which to perform a logical AND reduction. By default, a logical AND reduction must be
        performed over the entire array. If a tuple of integers, logical AND reductions must be performed over
        multiple axes. A valid axis must be an integer on the interval [-N, N), where N is the rank (number of
        dimensions) of x. If an axis is specified as a negative integer, the function must determine the axis along
        which to perform a reduction by counting backward from the last dimension (where -1 refers to the last
        dimension). If provided an invalid axis, the function must raise an exception. Default: None.
    keepdims : bool, optional
        If True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and,
        accordingly, the result must be compatible with the input array (see Broadcasting). Otherwise, if False,
        the reduced axes (dimensions) must not be included in the result. Default: False.

    Returns
    -------
    out : Array
        If a logical AND reduction was performed over the entire array, the returned array must be a zero-dimensional
        array containing the test result; otherwise, the returned array must be a non-zero-dimensional array
        containing the test results. The returned array must have a data type of bool.

    Note
    ----
    - Positive infinity, negative infinity, and NaN must evaluate to True.
    - If x has a complex floating-point data type, elements having a non-zero component (real or imaginary) must
    evaluate to True.
    - If x is an empty array or the size of the axis (dimension) along which to evaluate elements is zero, the test
    result must be True.
    """
    return NotImplemented


def any(x: Array, /, *, axis: None | int | tuple[int, ...] = None, keepdims: bool = False) -> Array:
    """
    Tests whether any input array element evaluates to True along a specified axis.

    Parameters
    ----------
    x : Array
        Input array.
    axis :  None | int | tuple[int, ...], optional
        Axis or axes along which to perform a logical OR reduction. By default, a logical OR reduction must be
        performed over the entire array. If a tuple of integers, logical OR reductions must be performed over
        multiple axes. A valid axis must be an integer on the interval [-N, N), where N is the rank (number of
        dimensions) of x. If an axis is specified as a negative integer, the function must determine the axis along
        which to perform a reduction by counting backward from the last dimension (where -1 refers to the last
        dimension). If provided an invalid axis, the function must raise an exception. Default: None.
    keepdims : bool, optional
        If True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and,
        accordingly, the result must be compatible with the input array (see Broadcasting). Otherwise, if False,
        the reduced axes (dimensions) must not be included in the result. Default: False.

    Returns
    -------
    out : Array
        If a logical OR reduction was performed over the entire array, the returned array must be a zero-dimensional
        array containing the test result; otherwise, the returned array must be a non-zero-dimensional array
        containing the test results. The returned array must have a data type of bool.

    Note
    ----
    - Positive infinity, negative infinity, and NaN must evaluate to True.
    - If x has a complex floating-point data type, elements having a non-zero component (real or imaginary) must
    evaluate to True.
    - If x is an empty array or the size of the axis (dimension) along which to evaluate elements is zero, the test
    result must be False.
    """
    return NotImplemented
