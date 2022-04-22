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

        print("Running LU InPlace\n")
        pivot = af.lu_inplace(in_array)
        print(in_array)
        print(pivot)

        print("Running LU with Upper Lower Factorization\n")
        lower, upper, pivot = af.lu(in_array)
        print(lower)
        print(upper)
        print(pivot)

    except Exception as e:
        print('Error: ', str(e))

if __name__ == '__main__':
    main()

