#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from .library import *
from .array import *

def lu(A):
    L = array()
    U = array()
    P = array()
    safe_call(clib.af_lu(pointer(L.arr), pointer(U.arr), pointer(P.arr), A.arr))
    return L,U,P

def lu_inplace(A, pivot="lapack"):
    P = array()
    is_pivot_lapack = False if (pivot == "full") else True
    safe_call(clib.af_lu_inplace(pointer(P.arr), A.arr, is_pivot_lapack))
    return P

def qr(A):
    Q = array()
    R = array()
    T = array()
    safe_call(clib.af_lu(pointer(Q.arr), pointer(R.arr), pointer(T.arr), A.arr))
    return Q,R,T

def qr_inplace(A):
    T = array()
    safe_call(clib.af_qr_inplace(pointer(T.arr), A.arr))
    return T

def cholesky(A, is_upper=True):
    R = array()
    info = c_int(0)
    safe_call(clib.af_cholesky(pointer(R.arr), pointer(info), A.arr, is_upper))
    return R, info.value

def cholesky_inplace(A, is_upper=True):
    info = c_int(0)
    safe_call(clib.af_cholesky_inplace(pointer(info), A.arr, is_upper))
    return info.value

def solve(A, B, options=AF_MAT_NONE):
    X = array()
    safe_call(clib.af_solve(pointer(X.arr), A.arr, B.arr, options))
    return X

def solve_lu(A, P, B, options=AF_MAT_NONE):
    X = array()
    safe_call(clib.af_solve_lu(pointer(X.arr), A.arr, P.arr, B.arr, options))
    return X

def inverse(A, options=AF_MAT_NONE):
    I = array()
    safe_call(clib.af_inverse(pointer(I.arr), A.arr, options))
    return I

def rank(A, tol=1E-5):
    r = c_uint(0)
    safe_call(clib.af_rank(pointer(r), A.arr, c_double(tol)))
    return r.value

def det(A):
    re = c_double(0)
    im = c_double(0)
    safe_call(clib.af_det(pointer(re), pointer(im), A.arr))
    re = re.value
    im = im.value
    return re if (im == 0) else re + im * 1j

def norm(A, norm_type=AF_NORM_EUCLID, p=1.0, q=1.0):
    res = c_double(0)
    safe_call(clib.af_norm(pointer(res), A.arr, norm_type, c_double(p), c_double(q)))
    return res.value
