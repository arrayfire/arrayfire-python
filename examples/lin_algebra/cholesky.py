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

        n = 5
        t = af.randu(n, n)
        arr_in = af.matmulNT(t, t) + af.identity(n, n) * n

        print("Running Cholesky InPlace\n")
        cin_upper = arr_in.copy()
        cin_lower = arr_in.copy()

        af.cholesky_inplace(cin_upper, True)
        af.cholesky_inplace(cin_lower, False)

        print(cin_upper)
        print(cin_lower)

        print("\nRunning Cholesky Out of place\n")

        out_upper, upper_success = af.cholesky(arr_in, True)
        out_lower, lower_success = af.cholesky(arr_in, False)

        if upper_success == 0:
            print(out_upper)
        if lower_success == 0:
            print(out_lower)

    except Exception as e:
        print('Error: ', str(e))

if __name__ == '__main__':
    main()
