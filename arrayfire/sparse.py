#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Functions to create and manipulate sparse matrices.
"""

from .library import *
from .array import *
import numbers
from .interop import to_array

__to_sparse_enum = [STORAGE.DENSE,
                    STORAGE.CSR,
                    STORAGE.CSC,
                    STORAGE.COO]


def create_sparse(values, row_idx, col_idx, nrows, ncols, storage = STORAGE.CSR):
    """
    Create a sparse matrix from it's constituent parts.

    Parameters
    ----------

    values : af.Array.
          - Contains the non zero elements of the sparse array.

    row_idx : af.Array.
          - Contains row indices of the sparse array.

    col_idx : af.Array.
          - Contains column indices of the sparse array.

    nrows   : int.
          - specifies the number of rows in sparse matrix.

    ncols   : int.
          - specifies the number of columns in sparse matrix.

    storage : optional: arrayfire.STORAGE. default: arrayfire.STORAGE.CSR.
          - Can be one of arrayfire.STORAGE.CSR, arrayfire.STORAGE.COO.

    Returns
    -------

    A sparse matrix.
    """
    assert(isinstance(values, Array))
    assert(isinstance(row_idx, Array))
    assert(isinstance(col_idx, Array))
    out = Array()
    safe_call(backend.get().af_create_sparse_array(c_pointer(out.arr), c_dim_t(nrows), c_dim_t(ncols),
                                                   values.arr, row_idx.arr, col_idx.arr, storage.value))
    return out

def create_sparse_from_host(values, row_idx, col_idx, nrows, ncols, storage = STORAGE.CSR):
    """
    Create a sparse matrix from it's constituent parts.

    Parameters
    ----------

    values : Any datatype that can be converted to array.
          - Contains the non zero elements of the sparse array.

    row_idx : Any datatype that can be converted to array.
          - Contains row indices of the sparse array.

    col_idx : Any datatype that can be converted to array.
          - Contains column indices of the sparse array.

    nrows   : int.
          - specifies the number of rows in sparse matrix.

    ncols   : int.
          - specifies the number of columns in sparse matrix.

    storage : optional: arrayfire.STORAGE. default: arrayfire.STORAGE.CSR.
          - Can be one of arrayfire.STORAGE.CSR, arrayfire.STORAGE.COO.

    Returns
    -------

    A sparse matrix.
    """
    return create_sparse(to_array(values), to_array(row_idx), to_array(col_idx), nrows, ncols, storage)

def create_sparse_from_dense(dense, storage = STORAGE.CSR):
    """
    Create a sparse matrix from a dense matrix.

    Parameters
    ----------

    dense : af.Array.
          - A dense matrix.

    storage : optional: arrayfire.STORAGE. default: arrayfire.STORAGE.CSR.
          - Can be one of arrayfire.STORAGE.CSR, arrayfire.STORAGE.COO.

    Returns
    -------

    A sparse matrix.
    """
    assert(isinstance(dense, Array))
    out = Array()
    safe_call(backend.get().af_create_sparse_array_from_dense(c_pointer(out.arr), dense.arr, storage.value))
    return out

def convert_sparse_to_dense(sparse):
    """
    Create a dense matrix from a sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
          - A sparse matrix.

    Returns
    -------

    A dense matrix.
    """
    out = Array()
    safe_call(backend.get().af_sparse_to_dense(c_pointer(out.arr), sparse.arr))
    return out

def sparse_get_info(sparse):
    """
    Get the constituent arrays and storage info from a sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
           - A sparse matrix.

    Returns
    --------
    (values, row_idx, col_idx, storage) where
    values : arrayfire.Array containing non zero elements from sparse matrix
    row_idx : arrayfire.Array containing the row indices
    col_idx : arrayfire.Array containing the column indices
    storage : sparse storage
    """
    values = Array()
    row_idx = Array()
    col_idx = Array()
    stype = c_int_t(0)
    safe_call(backend.get().af_sparse_get_info(c_pointer(values.arr), c_pointer(row_idx.arr),
                                               c_pointer(col_idx.arr), c_pointer(stype),
                                               sparse.arr))
    return (values, row_idx, col_idx, __to_sparse_enum[stype.value])

def sparse_get_values(sparse):
    """
    Get the non zero values from sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
           - A sparse matrix.

    Returns
    --------
    arrayfire array containing the non zero elements.

    """
    values = Array()
    safe_call(backend.get().af_sparse_get_values(c_pointer(values.arr), sparse.arr))
    return values

def sparse_get_row_idx(sparse):
    """
    Get the row indices from sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
           - A sparse matrix.

    Returns
    --------
    arrayfire array containing the non zero elements.

    """
    row_idx = Array()
    safe_call(backend.get().af_sparse_get_row_idx(c_pointer(row_idx.arr), sparse.arr))
    return row_idx

def sparse_get_col_idx(sparse):
    """
    Get the column indices from sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
           - A sparse matrix.

    Returns
    --------
    arrayfire array containing the non zero elements.

    """
    col_idx = Array()
    safe_call(backend.get().af_sparse_get_col_idx(c_pointer(col_idx.arr), sparse.arr))
    return col_idx

def sparse_get_nnz(sparse):
    """
    Get the column indices from sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
           - A sparse matrix.

    Returns
    --------
    Number of non zero elements in the sparse matrix.

    """
    nnz = c_dim_t(0)
    safe_call(backend.get().af_sparse_get_nnz(c_pointer(nnz), sparse.arr))
    return nnz.value

def sparse_get_storage(sparse):
    """
    Get the column indices from sparse matrix.

    Parameters
    ----------

    sparse : af.Array.
           - A sparse matrix.

    Returns
    --------
    Number of non zero elements in the sparse matrix.

    """
    storage = c_int_t(0)
    safe_call(backend.get().af_sparse_get_storage(c_pointer(storage), sparse.arr))
    return __to_sparse_enum[storage.value]

def convert_sparse(sparse, storage):
    """
    Convert sparse matrix from one format to another.

    Parameters
    ----------

    storage : arrayfire.STORAGE.

    Returns
    -------

    Sparse matrix converted to the appropriate type.
    """
    out = Array()
    safe_call(backend.get().af_sparse_convert_to(c_pointer(out.arr), sparse.arr, storage.value))
    return out
