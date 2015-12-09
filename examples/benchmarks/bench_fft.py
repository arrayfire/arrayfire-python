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
from arrayfire import (array, randu, matmul)
import arrayfire as af

def bench(A, iters = 100):
    start = time()
    for t in range(iters):
        B = af.fft2(A)
    af.sync()
    return (time() - start) / iters

if __name__ == "__main__":

    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))

    af.info()
    print("Benchmark N x N 2D fft")

    for M in range(7, 13):
        N = 1 << M
        A = af.randu(N, N)
        af.sync()

        t = bench(A)
        gflops = (10.0 * N * N * M) / (t * 1E9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (N, N, gflops))
