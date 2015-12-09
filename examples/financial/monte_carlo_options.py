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

def monte_carlo_options(N, K, t, vol, r, strike, steps, use_barrier = True, B = None, ty = af.Dtype.f32):
    payoff = af.constant(0, N, 1, dtype = ty)

    dt = t / float(steps - 1)
    s = af.constant(strike, N, 1, dtype = ty)

    randmat = af.randn(N, steps - 1, dtype = ty)
    randmat = af.exp((r - (vol * vol * 0.5)) * dt + vol * math.sqrt(dt) * randmat);

    S = af.product(af.join(1, s, randmat), 1)

    if (use_barrier):
        S = S * af.all_true(S < B, 1)

    payoff = af.maxof(0, S - K)
    return af.mean(payoff) * math.exp(-r * t)

def monte_carlo_simulate(N, use_barrier, num_iter = 10):
    steps = 180
    stock_price = 100.0
    maturity = 0.5
    volatility = 0.3
    rate = 0.01
    strike = 100
    barrier = 115.0

    start = time()
    for i in range(num_iter):
        monte_carlo_options(N, stock_price, maturity, volatility, rate, strike, steps,
                            use_barrier, barrier)

    return (time() - start) / num_iter

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    af.info()

    monte_carlo_simulate(1000, use_barrier = False)
    monte_carlo_simulate(1000, use_barrier = True )
    af.sync()

    for n in range(10000, 100001, 10000):
        print("Time for %7d paths - vanilla method: %4.3f ms, barrier method: % 4.3f ms\n" %
              (n, 1000 * monte_carlo_simulate(n, False, 100), 1000 * monte_carlo_simulate(n, True, 100)))
