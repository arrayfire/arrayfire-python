#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af
from time import time
import math
import sys

sqrt2 = math.sqrt(2.0)

def cnd(x):
    temp = (x > 0)
    return temp * (0.5 + af.erf(x/sqrt2)/2) + (1 - temp) * (0.5 - af.erf((-x)/sqrt2)/2)

def black_scholes(S, X, R, V, T):
    # S = Underlying stock price
    # X = Strike Price
    # R = Risk free rate of interest
    # V = Volatility
    # T = Time to maturity

    d1 = af.log(S / X)
    d1 = d1 + (R + (V * V) * 0.5) * T
    d1 = d1 / (V * af.sqrt(T))

    d2 = d1 - (V * af.sqrt(T))
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)

    C = S * cnd_d1 - (X * af.exp((-R) * T) * cnd_d2)
    P = X * af.exp((-R) * T) * (1 - cnd_d2) - (S * (1 -cnd_d1))

    return (C, P)

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    af.info()

    M = 4000

    S = af.randu(M, 1)
    X = af.randu(M, 1)
    R = af.randu(M, 1)
    V = af.randu(M, 1)
    T = af.randu(M, 1)

    (C, P) = black_scholes(S, X, R, V, T)
    af.eval(C)
    af.eval(P)
    af.sync()

    num_iter = 100
    for N in range(50, 501, 50):
        S = af.randu(M, N)
        X = af.randu(M, N)
        R = af.randu(M, N)
        V = af.randu(M, N)
        T = af.randu(M, N)
        af.sync()

        print("Input data size: %d elements" % (M * N))

        start = time()
        for i in range(num_iter):
            (C, P) = black_scholes(S, X, R, V, T)
            af.eval(C)
            af.eval(P)
        af.sync()
        sec = (time() - start) / num_iter

        print("Mean GPU Time: %0.6f ms\n\n" % (1000.0 * sec))
