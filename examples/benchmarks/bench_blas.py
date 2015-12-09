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
        B = af.matmul(A, A)
    af.sync()
    return (time() - start) / iters

if __name__ == "__main__":

    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))

    af.info()
    print("Benchmark N x N matrix multiply")

    for n in range(128, 2048 + 128, 128):
        A = af.randu(n, n)
        af.sync()

        t = bench(A)
        gflops = 2.0 * (n**3) / (t * 1E9)
        print("Time taken for %4d x %4d: %0.4f Gflops" % (n, n, gflops))
