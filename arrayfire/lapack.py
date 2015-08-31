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
    L = Array()
    U = Array()
    P = Array()
    safe_call(backend.get().af_lu(ct.pointer(L.arr), ct.pointer(U.arr), ct.pointer(P.arr), A.arr))
    return L,U,P

def lu_inplace(A, pivot="lapack"):
    P = Array()
    is_pivot_lapack = False if (pivot == "full") else True
    safe_call(backend.get().af_lu_inplace(ct.pointer(P.arr), A.arr, is_pivot_lapack))
    return P

def qr(A):
    Q = Array()
    R = Array()
    T = Array()
    safe_call(backend.get().af_lu(ct.pointer(Q.arr), ct.pointer(R.arr), ct.pointer(T.arr), A.arr))
    return Q,R,T

def qr_inplace(A):
    T = Array()
    safe_call(backend.get().af_qr_inplace(ct.pointer(T.arr), A.arr))
    return T

def cholesky(A, is_upper=True):
    R = Array()
    info = ct.c_int(0)
    safe_call(backend.get().af_cholesky(ct.pointer(R.arr), ct.pointer(info), A.arr, is_upper))
    return R, info.value

def cholesky_inplace(A, is_upper=True):
    info = ct.c_int(0)
    safe_call(backend.get().af_cholesky_inplace(ct.pointer(info), A.arr, is_upper))
    return info.value

def solve(A, B, options=AF_MAT_NONE):
    X = Array()
    safe_call(backend.get().af_solve(ct.pointer(X.arr), A.arr, B.arr, options))
    return X

def solve_lu(A, P, B, options=AF_MAT_NONE):
    X = Array()
    safe_call(backend.get().af_solve_lu(ct.pointer(X.arr), A.arr, P.arr, B.arr, options))
    return X

def inverse(A, options=AF_MAT_NONE):
    I = Array()
    safe_call(backend.get().af_inverse(ct.pointer(I.arr), A.arr, options))
    return I

def rank(A, tol=1E-5):
    r = ct.c_uint(0)
    safe_call(backend.get().af_rank(ct.pointer(r), A.arr, ct.c_double(tol)))
    return r.value

def det(A):
    re = ct.c_double(0)
    im = ct.c_double(0)
    safe_call(backend.get().af_det(ct.pointer(re), ct.pointer(im), A.arr))
    re = re.value
    im = im.value
    return re if (im == 0) else re + im * 1j

def norm(A, norm_type=AF_NORM_EUCLID, p=1.0, q=1.0):
    res = ct.c_double(0)
    safe_call(backend.get().af_norm(ct.pointer(res), A.arr, norm_type, ct.c_double(p), ct.c_double(q)))
    return res.value
