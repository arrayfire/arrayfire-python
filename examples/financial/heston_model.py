#!/usr/bin/python

##############################################################################################
# Copyright (c) 2015, Michael Nowotny
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################################

import arrayfire as af
import math
import time

def simulateHestonModel( T, N, R, mu, kappa, vBar, sigmaV, rho, x0, v0 ) :

    deltaT = T / (float)(N - 1)

    x = [af.constant(x0, R, dtype=af.Dtype.f32), af.constant(0, R, dtype=af.Dtype.f32)]
    v = [af.constant(v0, R, dtype=af.Dtype.f32), af.constant(0, R, dtype=af.Dtype.f32)]

    sqrtDeltaT = math.sqrt(deltaT)
    sqrtOneMinusRhoSquare = math.sqrt(1-rho**2)

    m = af.constant(0, 2, dtype=af.Dtype.f32)
    m[0] = rho
    m[1] = sqrtOneMinusRhoSquare
    zeroArray = af.constant(0, R, 1, dtype=af.Dtype.f32)

    for t in range(1, N) :
        tPrevious = (t + 1) % 2
        tCurrent =  t % 2

        dBt = af.randn(R, 2, dtype=af.Dtype.f32) * sqrtDeltaT

        vLag = af.maxof(v[tPrevious], zeroArray)
        sqrtVLag = af.sqrt(vLag)

        x[tCurrent] = x[tPrevious] + (mu - 0.5 * vLag) * deltaT + sqrtVLag * dBt[:, 0]
        v[tCurrent] = vLag + kappa * (vBar - vLag) * deltaT + sigmaV * (sqrtVLag * af.matmul(dBt, m))

    return (x[tCurrent], af.maxof(v[tCurrent], zeroArray))


def main():
    T = 1
    nT = 20 * T
    R_first = 1000
    R = 5000000

    x0 = 0 # initial log stock price
    v0 = 0.087**2 # initial volatility
    r = math.log(1.0319) # risk-free rate
    rho = -0.82 # instantaneous correlation between Brownian motions
    sigmaV = 0.14 # variance of volatility
    kappa = 3.46 # mean reversion speed
    vBar = 0.008 # mean variance
    k = math.log(0.95) # strike price

    # first run
    ( x, v ) = simulateHestonModel( T, nT, R_first, r, kappa, vBar, sigmaV, rho, x0, v0 )

    # Price plain vanilla call option
    tic = time.time()
    ( x, v ) = simulateHestonModel( T, nT, R, r, kappa, vBar, sigmaV, rho, x0, v0 )
    af.sync()
    toc = time.time() - tic
    K = math.exp(k)
    zeroConstant = af.constant(0, R, dtype=af.Dtype.f32)
    C_CPU = math.exp(-r * T) * af.mean(af.maxof(af.exp(x) - K, zeroConstant))
    print("Time elapsed = {} secs".format(toc))
    print("Call price = {}".format(C_CPU))
    print(af.mean(v))


if __name__ == "__main__":
    main()
