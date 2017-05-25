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


def calc_arrayfire(n):
    A = af.randu(n, n)
    af.sync()

    def run(iters):
        for t in range(iters):
            B = af.fft2(A)

        af.sync()

    return run


def calc_numpy(n):
    np.random.seed(1)
    A = np.random.rand(n, n).astype(np.float32)

    def run(iters):
        for t in range(iters):
            B = np.fft.fft2(A)

    return run


def bench(calc, iters=100, upto=13):
    _, name = calc.__name__.split("_")
    print("Benchmark N x N 2D fft on %s" % name)

    for M in range(7, upto):
        N = 1 << M
        run = calc(N)
        start = time()
        run(iters)
        t = (time() - start) / iters
        gflops = (10.0 * N * N * M) / (t * 1E9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (N, N, gflops))


if __name__ == "__main__":

    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))

    af.info()

    bench(calc_arrayfire)
    if np:
        bench(calc_numpy, upto=10)
