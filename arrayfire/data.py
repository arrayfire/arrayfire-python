#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Functions to create and manipulate arrays.
"""

from sys import version_info
from .library import *
from .array import *
from .util import *
from .util import _is_number
from .random import randu, randn, set_seed, get_seed

def constant(val, d0, d1=None, d2=None, d3=None, dtype=Dtype.f32):
    """
    Create a multi dimensional array whose elements contain the same value.

    Parameters
    ----------
    val : scalar.
          Value of each element of the constant array.

    d0 : int.
         Length of first dimension.

    d1 : optional: int. default: None.
         Length of second dimension.

    d2 : optional: int. default: None.
         Length of third dimension.

    d3 : optional: int. default: None.
         Length of fourth dimension.

    dtype : optional: af.Dtype. default: af.Dtype.f32.
           Data type of the array.

    Returns
    -------

    out : af.Array
          Multi dimensional array whose elements are of value `val`.
          - If d1 is None, `out` is 1D of size (d0,).
          - If d1 is not None and d2 is None, `out` is 2D of size (d0, d1).
          - If d1 and d2 are not None and d3 is None, `out` is 3D of size (d0, d1, d2).
          - If d1, d2, d3 are all not None, `out` is 4D of size (d0, d1, d2, d3).
    """

    out = Array()
    out.arr = constant_array(val, d0, d1, d2, d3, dtype.value)
    return out

# Store builtin range function to be used later
_brange = range

def range(d0, d1=None, d2=None, d3=None, dim=0, dtype=Dtype.f32):
    """
    Create a multi dimensional array using length of a dimension as range.

    Parameters
    ----------
    val : scalar.
          Value of each element of the constant array.

    d0 : int.
         Length of first dimension.

    d1 : optional: int. default: None.
         Length of second dimension.

    d2 : optional: int. default: None.
         Length of third dimension.

    d3 : optional: int. default: None.
         Length of fourth dimension.

    dim : optional: int. default: 0.
         The dimension along which the range is calculated.

    dtype : optional: af.Dtype. default: af.Dtype.f32.
           Data type of the array.

    Returns
    -------

    out : af.Array
          Multi dimensional array whose elements are along `dim` fall between [0 - self.dims[dim]-1]
          - If d1 is None, `out` is 1D of size (d0,).
          - If d1 is not None and d2 is None, `out` is 2D of size (d0, d1).
          - If d1 and d2 are not None and d3 is None, `out` is 3D of size (d0, d1, d2).
          - If d1, d2, d3 are all not None, `out` is 4D of size (d0, d1, d2, d3).


    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.range(3, 2) # dim is not specified, range is along first dimension.
    >>> af.display(a) # The data ranges from [0 - 2] (3 elements along first dimension)
    [3 2 1 1]
        0.0000     0.0000
        1.0000     1.0000
        2.0000     2.0000

    >>> a = af.range(3, 2, dim=1) # dim is 1, range is along second dimension.
    >>> af.display(a) # The data ranges from [0 - 1] (2 elements along second dimension)
    [3 2 1 1]
        0.0000     1.0000
        0.0000     1.0000
        0.0000     1.0000
    """
    out = Array()
    dims = dim4(d0, d1, d2, d3)

    safe_call(backend.get().af_range(c_pointer(out.arr), 4, c_pointer(dims), dim, dtype.value))
    return out


def iota(d0, d1=None, d2=None, d3=None, dim=-1, tile_dims=None, dtype=Dtype.f32):
    """
    Create a multi dimensional array using the number of elements in the array as the range.

    Parameters
    ----------
    val : scalar.
          Value of each element of the constant array.

    d0 : int.
         Length of first dimension.

    d1 : optional: int. default: None.
         Length of second dimension.

    d2 : optional: int. default: None.
         Length of third dimension.

    d3 : optional: int. default: None.
         Length of fourth dimension.

    tile_dims : optional: tuple of ints. default: None.
         The number of times the data is tiled.

    dtype : optional: af.Dtype. default: af.Dtype.f32.
           Data type of the array.

    Returns
    -------

    out : af.Array
          Multi dimensional array whose elements are along `dim` fall between [0 - self.elements() - 1].

    Examples
    --------
    >>> import arrayfire as af
    >>> import arrayfire as af
    >>> a = af.iota(3,3) # tile_dim is not specified, data is not tiled
    >>> af.display(a) # the elements range from [0 - 8] (9 elements)
    [3 3 1 1]
        0.0000     3.0000     6.0000
        1.0000     4.0000     7.0000
        2.0000     5.0000     8.0000

    >>> b = af.iota(3,3,tile_dims(1,2)) # Asking to tile along second dimension.
    >>> af.display(b)
    [3 6 1 1]
        0.0000     3.0000     6.0000     0.0000     3.0000     6.0000
        1.0000     4.0000     7.0000     1.0000     4.0000     7.0000
        2.0000     5.0000     8.0000     2.0000     5.0000     8.0000
    """
    out = Array()
    dims = dim4(d0, d1, d2, d3)
    td=[1]*4

    if tile_dims is not None:
        for i in _brange(len(tile_dims)):
            td[i] = tile_dims[i]

    tdims = dim4(td[0], td[1], td[2], td[3])

    safe_call(backend.get().af_iota(c_pointer(out.arr), 4, c_pointer(dims),
                                    4, c_pointer(tdims), dtype.value))
    return out

def identity(d0, d1, d2=None, d3=None, dtype=Dtype.f32):
    """
    Create an identity matrix or batch of identity matrices.

    Parameters
    ----------
    d0 : int.
         Length of first dimension.

    d1 : int.
         Length of second dimension.

    d2 : optional: int. default: None.
         Length of third dimension.

    d3 : optional: int. default: None.
         Length of fourth dimension.

    dtype : optional: af.Dtype. default: af.Dtype.f32.
           Data type of the array.

    Returns
    -------

    out : af.Array
          Multi dimensional array whose first two dimensions form a identity matrix.
          - If d2 is  None, `out` is 2D of size (d0, d1).
          - If d2 is not None and d3 is None, `out` is 3D of size (d0, d1, d2).
          - If d2, d3 are not None, `out` is 4D of size (d0, d1, d2, d3).
    """

    out = Array()
    dims = dim4(d0, d1, d2, d3)

    safe_call(backend.get().af_identity(c_pointer(out.arr), 4, c_pointer(dims), dtype.value))
    return out

def diag(a, num=0, extract=True):
    """
    Create a diagonal matrix or Extract the diagonal from a matrix.

    Parameters
    ----------
    a : af.Array.
        1 dimensional or 2 dimensional arrayfire array.

    num : optional: int. default: 0.
        The index of the diagonal.
        - num == 0 signifies the diagonal.
        - num  > 0 signifies super diagonals.
        - num <  0 signifies sub diagonals.

    extract : optional: bool. default: True.
         - If True , diagonal is extracted. `a` has to be 2D.
         - If False, diagonal matrix is created. `a` has to be 1D.

    Returns
    -------

    out : af.Array
         - if extract is True, `out` contains the num'th diagonal from `a`.
         - if extract is False, `out` contains `a` as the num'th diagonal.
    """
    out = Array()
    if extract:
        safe_call(backend.get().af_diag_extract(c_pointer(out.arr), a.arr, c_int_t(num)))
    else:
        safe_call(backend.get().af_diag_create(c_pointer(out.arr), a.arr, c_int_t(num)))
    return out

def join(dim, first, second, third=None, fourth=None):
    """
    Join two or more arrayfire arrays along a specified dimension.

    Parameters
    ----------

    dim: int.
        Dimension along which the join occurs.

    first : af.Array.
        Multi dimensional arrayfire array.

    second : af.Array.
        Multi dimensional arrayfire array.

    third : optional: af.Array. default: None.
        Multi dimensional arrayfire array.

    fourth : optional: af.Array. default: None.
        Multi dimensional arrayfire array.

    Returns
    -------

    out : af.Array
          An array containing the input arrays joined along the specified dimension.

    Examples
    ---------

    >>> import arrayfire as af
    >>> a = af.randu(2, 3)
    >>> b = af.randu(2, 3)
    >>> c = af.join(0, a, b)
    >>> d = af.join(1, a, b)
    >>> af.display(a)
    [2 3 1 1]
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719

    >>> af.display(b)
    [2 3 1 1]
        0.3266     0.6009     0.2442
        0.6275     0.0495     0.6591

    >>> af.display(c)
    [4 3 1 1]
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719
        0.3266     0.6009     0.2442
        0.6275     0.0495     0.6591

    >>> af.display(d)
    [2 6 1 1]
        0.9508     0.2591     0.7928     0.3266     0.6009     0.2442
        0.5367     0.8359     0.8719     0.6275     0.0495     0.6591
    """
    out = Array()
    if (third is None and fourth is None):
        safe_call(backend.get().af_join(c_pointer(out.arr), dim, first.arr, second.arr))
    else:
        c_void_p_4 = c_void_ptr_t * 4
        c_array_vec = c_void_p_4(first.arr, second.arr, 0, 0)
        num = 2
        if third is not None:
            c_array_vec[num] = third.arr
            num+=1
        if fourth is not None:
            c_array_vec[num] = fourth.arr
            num+=1

        safe_call(backend.get().af_join_many(c_pointer(out.arr), dim, num, c_pointer(c_array_vec)))
    return out


def tile(a, d0, d1=1, d2=1, d3=1):
    """
    Tile an array along specified dimensions.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    d0: int.
        The number of times `a` has to be tiled along first dimension.

    d1: optional: int. default: 1.
        The number of times `a` has to be tiled along second dimension.

    d2: optional: int. default: 1.
        The number of times `a` has to be tiled along third dimension.

    d3: optional: int. default: 1.
        The number of times `a` has to be tiled along fourth dimension.

    Returns
    -------

    out : af.Array
          An array containing the input after tiling the the specified number of times.

    Examples
    ---------

    >>> import arrayfire as af
    >>> a = af.randu(2, 3)
    >>> b = af.tile(a, 2)
    >>> c = af.tile(a, 1, 2)
    >>> d = af.tile(a, 2, 2)
    >>> af.display(a)
    [2 3 1 1]
        0.9508     0.2591     0.7928
        0.5367     0.8359     0.8719

    >>> af.display(b)
    [4 3 1 1]
        0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081
        0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081

    >>> af.display(c)
    [2 6 1 1]
        0.4107     0.9518     0.4198     0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081     0.8224     0.1794     0.0081

    >>> af.display(d)
    [4 6 1 1]
        0.4107     0.9518     0.4198     0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081     0.8224     0.1794     0.0081
        0.4107     0.9518     0.4198     0.4107     0.9518     0.4198
        0.8224     0.1794     0.0081     0.8224     0.1794     0.0081
    """
    out = Array()
    safe_call(backend.get().af_tile(c_pointer(out.arr), a.arr, d0, d1, d2, d3))
    return out

def reorder(a, d0=1, d1=0, d2=2, d3=3):
    """
    Reorder the dimensions of the input.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    d0: optional: int. default: 1.
        The location of the first dimension in the output.

    d1: optional: int. default: 0.
        The location of the second dimension in the output.

    d2: optional: int. default: 2.
        The location of the third dimension in the output.

    d3: optional: int. default: 3.
        The location of the fourth dimension in the output.

    Returns
    -------

    out : af.Array
          - An array containing the input aftern reordering its dimensions.

    Note
    ------
    - `af.reorder(a, 1, 0)` is the same as `transpose(a)`

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(5, 5, 3)
    >>> af.display(a)
    [5 5 3 1]
        0.4107     0.0081     0.6600     0.1046     0.8395
        0.8224     0.3775     0.0764     0.8827     0.1933
        0.9518     0.3027     0.0901     0.1647     0.7270
        0.1794     0.6456     0.5933     0.8060     0.0322
        0.4198     0.5591     0.1098     0.5938     0.0012

        0.8703     0.9250     0.4387     0.6530     0.4224
        0.5259     0.3063     0.3784     0.5476     0.5293
        0.1443     0.9313     0.4002     0.8577     0.0212
        0.3253     0.8684     0.4390     0.8370     0.1103
        0.5081     0.6592     0.4718     0.0618     0.4420

        0.8355     0.6767     0.1033     0.9426     0.9276
        0.4878     0.6742     0.2119     0.4817     0.8662
        0.2055     0.4523     0.5955     0.9097     0.3578
        0.1794     0.1236     0.3745     0.6821     0.6263
        0.5606     0.7924     0.9165     0.6056     0.9747


    >>> b = af.reorder(a, 2, 0, 1)
    >>> af.display(b)
    [3 5 5 1]
        0.4107     0.8224     0.9518     0.1794     0.4198
        0.8703     0.5259     0.1443     0.3253     0.5081
        0.8355     0.4878     0.2055     0.1794     0.5606

        0.0081     0.3775     0.3027     0.6456     0.5591
        0.9250     0.3063     0.9313     0.8684     0.6592
        0.6767     0.6742     0.4523     0.1236     0.7924

        0.6600     0.0764     0.0901     0.5933     0.1098
        0.4387     0.3784     0.4002     0.4390     0.4718
        0.1033     0.2119     0.5955     0.3745     0.9165

        0.1046     0.8827     0.1647     0.8060     0.5938
        0.6530     0.5476     0.8577     0.8370     0.0618
        0.9426     0.4817     0.9097     0.6821     0.6056

        0.8395     0.1933     0.7270     0.0322     0.0012
        0.4224     0.5293     0.0212     0.1103     0.4420
        0.9276     0.8662     0.3578     0.6263     0.9747
    """
    out = Array()
    safe_call(backend.get().af_reorder(c_pointer(out.arr), a.arr, d0, d1, d2, d3))
    return out

def shift(a, d0, d1=0, d2=0, d3=0):
    """
    Shift the input along each dimension.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    d0: int.
        The amount of shift along first dimension.

    d1: optional: int. default: 0.
        The amount of shift along second dimension.

    d2: optional: int. default: 0.
        The amount of shift along third dimension.

    d3: optional: int. default: 0.
        The amount of shift along fourth dimension.

    Returns
    -------

    out : af.Array
          - An array the same shape as `a` after shifting it by the specified amounts.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(3, 3)
    >>> b = af.shift(a, 2)
    >>> c = af.shift(a, 1, -1)
    >>> af.display(a)
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> af.display(b)
    [3 3 1 1]
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363
        0.7269     0.3569     0.3341

    >>> af.display(c)
    [3 3 1 1]
        0.4563     0.5363     0.5201
        0.3569     0.3341     0.7269
        0.1437     0.0899     0.7104
    """
    out = Array()
    safe_call(backend.get().af_shift(c_pointer(out.arr), a.arr, d0, d1, d2, d3))
    return out

def moddims(a, d0, d1=1, d2=1, d3=1):
    """
    Modify the shape of the array without changing the data layout.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    d0: int.
        The first dimension of output.

    d1: optional: int. default: 1.
        The second dimension of output.

    d2: optional: int. default: 1.
        The third dimension of output.

    d3: optional: int. default: 1.
        The fourth dimension of output.

    Returns
    -------

    out : af.Array
          - An containing the same data as `a` with the specified shape.
          - The number of elements in `a` must match `d0 x d1 x d2 x d3`.
    """
    out = Array()
    dims = dim4(d0, d1, d2, d3)
    safe_call(backend.get().af_moddims(c_pointer(out.arr), a.arr, 4, c_pointer(dims)))
    return out

def flat(a):
    """
    Flatten the input array.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    Returns
    -------

    out : af.Array
          - 1 dimensional array containing all the elements from `a`.
    """
    out = Array()
    safe_call(backend.get().af_flat(c_pointer(out.arr), a.arr))
    return out

def flip(a, dim=0):
    """
    Flip an array along a dimension.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    dim : optional: int. default: 0.
       The dimension along which the flip is performed.

    Returns
    -------

    out : af.Array
          The output after flipping `a` along `dim`.

    Examples
    ---------

    >>> import arrayfire as af
    >>> a = af.randu(3, 3)
    >>> af.display(a)
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> af.display(b)
    [3 3 1 1]
        0.5201     0.4563     0.5363
        0.7104     0.1437     0.0899
        0.7269     0.3569     0.3341

    >>> af.display(c)
    [3 3 1 1]
        0.3341     0.3569     0.7269
        0.0899     0.1437     0.7104
        0.5363     0.4563     0.5201

    """
    out = Array()
    safe_call(backend.get().af_flip(c_pointer(out.arr), a.arr, c_int_t(dim)))
    return out

def lower(a, is_unit_diag=False):
    """
    Extract the lower triangular matrix from the input.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    is_unit_diag: optional: bool. default: False.
       Flag specifying if the diagonal elements are 1.

    Returns
    -------

    out : af.Array
          An array containing the lower triangular elements from `a`.
    """
    out = Array()
    safe_call(backend.get().af_lower(c_pointer(out.arr), a.arr, is_unit_diag))
    return out

def upper(a, is_unit_diag=False):
    """
    Extract the upper triangular matrix from the input.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    is_unit_diag: optional: bool. default: False.
       Flag specifying if the diagonal elements are 1.

    Returns
    -------

    out : af.Array
          An array containing the upper triangular elements from `a`.
    """
    out = Array()
    safe_call(backend.get().af_upper(c_pointer(out.arr), a.arr, is_unit_diag))
    return out

def select(cond, lhs, rhs):
    """
    Select elements from one of two arrays based on condition.

    Parameters
    ----------

    cond : af.Array
           Conditional array

    lhs  : af.Array or scalar
           numerical array whose elements are picked when conditional element is True

    rhs  : af.Array or scalar
           numerical array whose elements are picked when conditional element is False

    Returns
    --------

    out: af.Array
         An array containing elements from `lhs` when `cond` is True and `rhs` when False.

    Examples
    ---------

    >>> import arrayfire as af
    >>> a = af.randu(3,3)
    >>> b = af.randu(3,3)
    >>> cond = a > b
    >>> res = af.select(cond, a, b)

    >>> af.display(a)
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> af.display(b)
    [3 3 1 1]
        0.7269     0.3569     0.3341
        0.7104     0.1437     0.0899
        0.5201     0.4563     0.5363

    >>> af.display(res)
    [3 3 1 1]
        0.7269     0.3569     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.4563     0.6456
    """
    out = Array()

    is_left_array = isinstance(lhs, Array)
    is_right_array = isinstance(rhs, Array)

    if not (is_left_array or is_right_array):
        raise TypeError("Atleast one input needs to be of type arrayfire.array")

    elif (is_left_array and is_right_array):
        safe_call(backend.get().af_select(c_pointer(out.arr), cond.arr, lhs.arr, rhs.arr))

    elif (_is_number(rhs)):
        safe_call(backend.get().af_select_scalar_r(c_pointer(out.arr), cond.arr, lhs.arr, c_double_t(rhs)))
    else:
        safe_call(backend.get().af_select_scalar_l(c_pointer(out.arr), cond.arr, c_double_t(lhs), rhs.arr))

    return out

def replace(lhs, cond, rhs):
    """
    Select elements from one of two arrays based on condition.

    Parameters
    ----------

    lhs  : af.Array or scalar
           numerical array whose elements are replaced with `rhs` when conditional element is False

    cond : af.Array
           Conditional array

    rhs  : af.Array or scalar
           numerical array whose elements are picked when conditional element is False

    Examples
    ---------
    >>> import arrayfire as af
    >>> a = af.randu(3,3)
    >>> af.display(a)
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> cond = (a >= 0.25) & (a <= 0.75)
    >>> af.display(cond)
    [3 3 1 1]
             1          0          1
             0          1          1
             0          0          1

    >>> af.replace(a, cond, 0.3333)
    >>> af.display(a)
    [3 3 1 1]
        0.3333     0.1794     0.3333
        0.8224     0.3333     0.3333
        0.9518     0.0081     0.3333

    """
    is_right_array = isinstance(rhs, Array)

    if (is_right_array):
        safe_call(backend.get().af_replace(lhs.arr, cond.arr, rhs.arr))
    else:
        safe_call(backend.get().af_replace_scalar(lhs.arr, cond.arr, c_double_t(rhs)))

def lookup(a, idx, dim=0):
    """
    Lookup the values of input array based on index.

    Parameters
    ----------

    a : af.Array.
       Multi dimensional array.

    idx : is lookup indices

    dim : optional: int. default: 0.
       Specifies the dimension for indexing

    Returns
    -------

    out : af.Array
          An array containing values at locations specified by 'idx'

    Examples
    ---------

    >>> import arrayfire as af
    >>> arr = af.Array([1,0,3,4,5,6], (2,3))
    >>> af.display(arr)
    [2 3 1 1]
        1.0000     3.0000     5.0000
        0.0000     4.0000     6.0000

    >>> idx = af.array([0, 2])
    >>> af.lookup(arr, idx, 1)
    [2 2 1 1]
        1.0000     5.0000
        0.0000     6.0000

    >>> idx = af.array([0])
    >>> af.lookup(arr, idx, 0)
    [2 1 1 1]
        0.0000
        2.0000
    """
    out = Array()
    safe_call(backend.get().af_lookup(c_pointer(out.arr), a.arr, idx.arr, c_int_t(dim)))
    return out
