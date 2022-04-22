#!/usr/bin/env python
#######################################################
# Copyright (c) 2022, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af

def main():
    try:
        af.info()
        in_array = af.randu(5,8)

        print("Running QR InPlace\n")
        q_in = in_array.copy()
        print(q_in)

        tau = af.qr_inplace(q_in)

        print(q_in)
        print(tau)

        print("Running QR with Q and R factorization\n")
        q, r, tau = af.qr(in_array)

        print(q)
        print(r)
        print(tau)

    except Exception as e:
        print("Error: ", str(e))

if __name__ == '__main__':
    main()
