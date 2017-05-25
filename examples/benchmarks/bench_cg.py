#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################


import sys
from time import time
import arrayfire as af

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import sparse as sp
    from scipy.sparse import linalg
except ImportError:
    sp = None


def to_numpy(A):
    return np.asarray(A.to_list(), dtype=np.float32)


def to_sparse(A):
    return af.sparse.create_sparse_from_dense(A)


def to_scipy_sparse(spA, fmt='csr'):
    vals = np.asarray(af.sparse.sparse_get_values(spA).to_list(),
                      dtype = np.float32)
    rows = np.asarray(af.sparse.sparse_get_row_idx(spA).to_list(),
                      dtype = np.int)
    cols = np.asarray(af.sparse.sparse_get_col_idx(spA).to_list(),
                      dtype = np.int)
    return sp.csr_matrix((vals, cols, rows), dtype=np.float32)


def setup_input(n, sparsity=7):
    T = af.randu(n, n, dtype=af.Dtype.f32)
    A = af.floor(T*1000)
    A = A * ((A % sparsity) == 0) / 1000
    A = A.T + A + n*af.identity(n, n, dtype=af.Dtype.f32)
    x0 = af.randu(n, dtype=af.Dtype.f32)
    b = af.matmul(A, x0)
    # printing
    # nnz = af.sum((A != 0))
    # print "Sparsity of A: %2.2f %%" %(100*nnz/n**2,)
    return A, b, x0


def input_info(A, Asp):
    m, n = A.dims()
    nnz = af.sum((A != 0))
    print("    matrix size:                %i x %i" %(m, n))
    print("    matrix sparsity:            %2.2f %%" %(100*nnz/n**2,))
    print("    dense matrix memory usage:  ")
    print("    sparse matrix memory usage: ")


def calc_arrayfire(A, b, x0, maxiter=10):
    x = af.constant(0, b.dims()[0], dtype=af.Dtype.f32)
    r = b - af.matmul(A, x)
    p = r
    for i in range(maxiter):
        Ap = af.matmul(A, p)
        alpha_num = af.dot(r, r)
        alpha_den = af.dot(p, Ap)
        alpha = alpha_num/alpha_den
        r -= af.tile(alpha, Ap.dims()[0]) * Ap
        x += af.tile(alpha, Ap.dims()[0]) * p
        beta_num = af.dot(r, r)
        beta = beta_num/alpha_num
        p = r + af.tile(beta, p.dims()[0]) * p
    res = x0 - x
    return x, af.dot(res, res)


def calc_numpy(A, b, x0, maxiter=10):
    x = np.zeros(len(b), dtype=np.float32)
    r = b - np.dot(A, x)
    p = r.copy()
    for i in range(maxiter):
        Ap = np.dot(A, p)
        alpha_num = np.dot(r, r)
        alpha_den = np.dot(p, Ap)
        alpha = alpha_num/alpha_den
        r -= alpha * Ap
        x += alpha * p
        beta_num = np.dot(r, r)
        beta = beta_num/alpha_num
        p = r + beta * p
    res = x0 - x
    return x, np.dot(res, res)


def calc_scipy_sparse(A, b, x0, maxiter=10):
    x = np.zeros(len(b), dtype=np.float32)
    r = b - A*x
    p = r.copy()
    for i in range(maxiter):
        Ap = A*p
        alpha_num = np.dot(r, r)
        alpha_den = np.dot(p, Ap)
        alpha = alpha_num/alpha_den
        r -= alpha * Ap
        x += alpha * p
        beta_num = np.dot(r, r)
        beta = beta_num/alpha_num
        p = r + beta * p
    res = x0 - x
    return x, np.dot(res, res)


def calc_scipy_sparse_linalg_cg(A, b, x0, maxiter=10):
    x = np.zeros(len(b), dtype=np.float32)
    x, _ = linalg.cg(A, b, x, tol=0., maxiter=maxiter)
    res = x0 - x
    return x, np.dot(res, res)


def timeit(calc, iters, args):
    t0 = time()
    for i in range(iters):
        calc(*args)
    dt = time() - t0
    return 1000*dt/iters  # ms


def test():
    print("\nTesting benchmark functions...")
    A, b, x0 = setup_input(50)  # dense A
    Asp = to_sparse(A)
    x1, _ = calc_arrayfire(A, b, x0)
    x2, _ = calc_arrayfire(Asp, b, x0)
    if af.sum(af.abs(x1 - x2)/x2 > 1e-6):
        raise ValueError("arrayfire test failed")
    if np:
        An = to_numpy(A)
        bn = to_numpy(b)
        x0n = to_numpy(x0)
        x3, _ = calc_numpy(An, bn, x0n)
        if not np.allclose(x3, x1.to_list()):
            raise ValueError("numpy test failed")
    if sp:
        Asc = to_scipy_sparse(Asp)
        x4, _ = calc_scipy_sparse(Asc, bn, x0n)
        if not np.allclose(x4, x1.to_list()):
            raise ValueError("scipy.sparse test failed")
        x5, _ = calc_scipy_sparse_linalg_cg(Asc, bn, x0n)
        if not np.allclose(x5, x1.to_list()):
            raise ValueError("scipy.sparse.linalg.cg test failed")
    print("    all tests passed...")


def bench(n=4*1024, sparsity=7, maxiter=10, iters=10):
    # generate data
    print("\nGenerating benchmark data for n = %i ..." %n)
    A, b, x0 = setup_input(n, sparsity)  # dense A
    Asp = to_sparse(A)  # sparse A
    input_info(A, Asp)
    # make benchmarks
    print("Benchmarking CG solver for n = %i ..." %n)
    t1 = timeit(calc_arrayfire, iters, args=(A, b, x0, maxiter))
    print("    arrayfire - dense:            %f ms" %t1)
    t2 = timeit(calc_arrayfire, iters, args=(Asp, b, x0, maxiter))
    print("    arrayfire - sparse:           %f ms" %t2)
    if np:
        An = to_numpy(A)
        bn = to_numpy(b)
        x0n = to_numpy(x0)
        t3 = timeit(calc_numpy, iters, args=(An, bn, x0n, maxiter))
        print("    numpy     - dense:            %f ms" %t3)
    if sp:
        Asc = to_scipy_sparse(Asp)
        t4 = timeit(calc_scipy_sparse, iters, args=(Asc, bn, x0n, maxiter))
        print("    scipy     - sparse:           %f ms" %t4)
        t5 = timeit(calc_scipy_sparse_linalg_cg, iters, args=(Asc, bn, x0n, maxiter))
        print("    scipy     - sparse.linalg.cg: %f ms" %t5)

if __name__ == "__main__":
    #af.set_backend('cpu', unsafe=True)

    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))

    af.info()    

    test()
    
    for n in (128, 256, 512, 1024, 2048, 4096):
        bench(n)
