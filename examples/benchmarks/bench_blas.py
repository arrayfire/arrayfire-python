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
            B = af.matmul(A, A)
        af.sync()

    return run


def calc_numpy(n):
    np.random.seed(1)
    A = np.random.rand(n, n).astype(np.float32)

    def run(iters):
        for t in range(iters):
            B = np.dot(A, A)

    return run


def bench(calc, iters=100, upto=2048):
    _, name = calc.__name__.split("_")
    print("Benchmark N x N matrix multiply on %s" % name)

    for n in range(128, upto + 128, 128):
        run = calc(n)
        start = time()
        run(iters)
        t = (time() - start) / iters
        gflops = 2.0 * (n ** 3) / (t * 1E9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (n, n, gflops))


if __name__ == "__main__":

    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))

    af.info()

    bench(calc_arrayfire)
    if np:
        bench(calc_numpy, upto=512)
