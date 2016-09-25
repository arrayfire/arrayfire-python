#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Dense Linear Algebra functions (solve, inverse, etc).
"""

from .library import *
from .array import *

def lu(A):
    """
    LU decomposition.

    Parameters
    ----------
    A: af.Array
       A 2 dimensional arrayfire array.

    Returns
    -------
    (L,U,P): tuple of af.Arrays
           - L - Lower triangular matrix.
           - U - Upper triangular matrix.
           - P - Permutation array.

    Note
    ----

    The original matrix `A` can be reconstructed using the outputs in the following manner.

    >>> A[P, :] = af.matmul(L, U)

    """
    L = Array()
    U = Array()
    P = Array()
    safe_call(backend.get().af_lu(c_pointer(L.arr), c_pointer(U.arr), c_pointer(P.arr), A.arr))
    return L,U,P

def lu_inplace(A, pivot="lapack"):
    """
    In place LU decomposition.

    Parameters
    ----------
    A: af.Array
       - a 2 dimensional arrayfire array on entry.
       - Contains L in the lower triangle on exit.
       - Contains U in the upper triangle on exit.

    Returns
    -------
    P: af.Array
       - Permutation array.

    Note
    ----

    This function is primarily used with `af.solve_lu` to reduce computations.

    """
    P = Array()
    is_pivot_lapack = False if (pivot == "full") else True
    safe_call(backend.get().af_lu_inplace(c_pointer(P.arr), A.arr, is_pivot_lapack))
    return P

def qr(A):
    """
    QR decomposition.

    Parameters
    ----------
    A: af.Array
       A 2 dimensional arrayfire array.

    Returns
    -------
    (Q,R,T): tuple of af.Arrays
           - Q - Orthogonal matrix.
           - R - Upper triangular matrix.
           - T - Vector containing additional information to solve a least squares problem.

    Note
    ----

    The outputs of this funciton have the following properties.

    >>> A = af.matmul(Q, R)
    >>> I = af.matmulNT(Q, Q) # Identity matrix
    """
    Q = Array()
    R = Array()
    T = Array()
    safe_call(backend.get().af_qr(c_pointer(Q.arr), c_pointer(R.arr), c_pointer(T.arr), A.arr))
    return Q,R,T

def qr_inplace(A):
    """
    In place QR decomposition.

    Parameters
    ----------
    A: af.Array
       - a 2 dimensional arrayfire array on entry.
       - Packed Q and R matrices on exit.

    Returns
    -------
    T: af.Array
       - Vector containing additional information to solve a least squares problem.

    Note
    ----

    This function is used to save space only when `R` is required.
    """
    T = Array()
    safe_call(backend.get().af_qr_inplace(c_pointer(T.arr), A.arr))
    return T

def cholesky(A, is_upper=True):
    """
    Cholesky decomposition

    Parameters
    ----------
    A: af.Array
       A 2 dimensional, symmetric, positive definite matrix.

    is_upper: optional: bool. default: True
       Specifies if output `R` is upper triangular (if True) or lower triangular (if False).

    Returns
    -------
    (R,info): tuple of af.Array, int.
           - R - triangular matrix.
           - info - 0 if decomposition sucessful.

    Note
    ----

    The original matrix `A` can be reconstructed using the outputs in the following manner.

    >>> A = af.matmulNT(R, R) #if R is upper triangular

    """
    R = Array()
    info = c_int_t(0)
    safe_call(backend.get().af_cholesky(c_pointer(R.arr), c_pointer(info), A.arr, is_upper))
    return R, info.value

def cholesky_inplace(A, is_upper=True):
    """
    In place Cholesky decomposition.

    Parameters
    ----------
    A: af.Array
       - a 2 dimensional, symmetric, positive definite matrix.
       - Trinangular matrix on exit.

    is_upper: optional: bool. default: True.
       Specifies if output `R` is upper triangular (if True) or lower triangular (if False).

    Returns
    -------
    info : int.
           0 if decomposition sucessful.

    """
    info = c_int_t(0)
    safe_call(backend.get().af_cholesky_inplace(c_pointer(info), A.arr, is_upper))
    return info.value

def solve(A, B, options=MATPROP.NONE):
    """
    Solve a system of linear equations.

    Parameters
    ----------

    A: af.Array
       A 2 dimensional arrayfire array representing the coefficients of the system.

    B: af.Array
       A 1 or 2 dimensional arrayfire array representing the constants of the system.

    options: optional: af.MATPROP. default: af.MATPROP.NONE.
       - Additional options to speed up computations.
       - Currently needs to be one of `af.MATPROP.NONE`, `af.MATPROP.LOWER`, `af.MATPROP.UPPER`.

    Returns
    -------
    X: af.Array
       A 1 or 2 dimensional arrayfire array representing the unknowns in the system.

    """
    X = Array()
    safe_call(backend.get().af_solve(c_pointer(X.arr), A.arr, B.arr, options.value))
    return X

def solve_lu(A, P, B, options=MATPROP.NONE):
    """
    Solve a system of linear equations, using LU decomposition.

    Parameters
    ----------

    A: af.Array
       - A 2 dimensional arrayfire array representing the coefficients of the system.
       - This matrix should be decomposed previously using `lu_inplace(A)`.

    P: af.Array
       - Permutation array.
       - This array is the output of an earlier call to `lu_inplace(A)`

    B: af.Array
       A 1 or 2 dimensional arrayfire array representing the constants of the system.

    Returns
    -------
    X: af.Array
       A 1 or 2 dimensional arrayfire array representing the unknowns in the system.

    """
    X = Array()
    safe_call(backend.get().af_solve_lu(c_pointer(X.arr), A.arr, P.arr, B.arr, options.value))
    return X

def inverse(A, options=MATPROP.NONE):
    """
    Invert a matrix.

    Parameters
    ----------

    A: af.Array
       - A 2 dimensional arrayfire array

    options: optional: af.MATPROP. default: af.MATPROP.NONE.
       - Additional options to speed up computations.
       - Currently needs to be one of `af.MATPROP.NONE`.

    Returns
    -------

    AI: af.Array
       - A 2 dimensional array that is the inverse of `A`

    Note
    ----

    `A` needs to be a square matrix.

    """
    AI = Array()
    safe_call(backend.get().af_inverse(c_pointer(AI.arr), A.arr, options.value))
    return AI

def rank(A, tol=1E-5):
    """
    Rank of a matrix.

    Parameters
    ----------

    A: af.Array
       - A 2 dimensional arrayfire array

    tol: optional: scalar. default: 1E-5.
       - Tolerance for calculating rank

    Returns
    -------

    r: int
       - Rank of `A` within the given tolerance
    """
    r = c_uint_t(0)
    safe_call(backend.get().af_rank(c_pointer(r), A.arr, c_double_t(tol)))
    return r.value

def det(A):
    """
    Determinant of a matrix.

    Parameters
    ----------

    A: af.Array
       - A 2 dimensional arrayfire array

    Returns
    -------

    res: scalar
       - Determinant of the matrix.
    """
    re = c_double_t(0)
    im = c_double_t(0)
    safe_call(backend.get().af_det(c_pointer(re), c_pointer(im), A.arr))
    re = re.value
    im = im.value
    return re if (im == 0) else re + im * 1j

def norm(A, norm_type=NORM.EUCLID, p=1.0, q=1.0):
    """
    Norm of an array or a matrix.

    Parameters
    ----------

    A: af.Array
       - A 1 or 2 dimensional arrayfire array

    norm_type: optional: af.NORM. default: af.NORM.EUCLID.
       - Type of norm to be calculated.

    p: scalar. default 1.0.
       - Used only if `norm_type` is one of `af.NORM.VECTOR_P`, `af.NORM_MATRIX_L_PQ`

    q: scalar. default 1.0.
       - Used only if `norm_type` is `af.NORM_MATRIX_L_PQ`

    Returns
    -------

    res: scalar
       - norm of the input

    """
    res = c_double_t(0)
    safe_call(backend.get().af_norm(c_pointer(res), A.arr, norm_type.value,
                                    c_double_t(p), c_double_t(q)))
    return res.value

def svd(A):
    """
    Singular Value Decomposition

    Parameters
    ----------
    A: af.Array
       A 2 dimensional arrayfire array.

    Returns
    -------
    (U,S,Vt): tuple of af.Arrays
           - U - A unitary matrix
           - S - An array containing the elements of diagonal matrix
           - Vt - A unitary matrix

    Note
    ----

    - The original matrix `A` is preserved and additional storage space is required for decomposition.

    - If the original matrix `A` need not be preserved, use `svd_inplace` instead.

    - The original matrix `A` can be reconstructed using the outputs in the following manner.
    >>> Smat = af.diag(S, 0, False)
    >>> A_recon = af.matmul(af.matmul(U, Smat), Vt)

    """
    U = Array()
    S = Array()
    Vt = Array()
    safe_call(backend.get().af_svd(c_pointer(U.arr), c_pointer(S.arr), c_pointer(Vt.arr), A.arr))
    return U, S, Vt

def svd_inplace(A):
    """
    Singular Value Decomposition

    Parameters
    ----------
    A: af.Array
       A 2 dimensional arrayfire array.

    Returns
    -------
    (U,S,Vt): tuple of af.Arrays
           - U - A unitary matrix
           - S - An array containing the elements of diagonal matrix
           - Vt - A unitary matrix

    Note
    ----

    - The original matrix `A` is not preserved.

    - If the original matrix `A` needs to be preserved, use `svd` instead.

    - The original matrix `A` can be reconstructed using the outputs in the following manner.
    >>> Smat = af.diag(S, 0, False)
    >>> A_recon = af.matmul(af.matmul(U, Smat), Vt)

    """
    U = Array()
    S = Array()
    Vt = Array()
    safe_call(backend.get().af_svd_inplace(c_pointer(U.arr), c_pointer(S.arr), c_pointer(Vt.arr),
                                           A.arr))
    return U, S, Vt

def is_lapack_available():
    """
    Function to check if the arrayfire library was built with lapack support.
    """
    res = c_bool_t(False)
    safe_call(backend.get().af_is_lapack_available(c_pointer(res)))
    return res.value
