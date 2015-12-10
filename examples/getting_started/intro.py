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
import sys
from array import array

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    af.info()

    print("\n---- Intro to ArrayFire using signed(s32) arrays ----\n")

    h_A = array('i', ( 1,  2,  4, -1,  2,  0,  4,  2,  3))
    h_B = array('i', ( 2,  3,  5,  6,  0, 10,-12,  0,  1))

    A = af.Array(h_A, (3,3))
    B = af.Array(h_B, (3,3))

    print("\n---- Sub referencing and sub assignment\n")
    af.display(A)
    af.display(A[0,:])
    af.display(A[:,0])
    A[0,0] = 11
    A[1] = 100
    af.display(A)
    af.display(B)
    A[1,:] = B[2,:]
    af.display(A)

    print("\n---- Bitwise operations\n")
    af.display(A & B)
    af.display(A | B)
    af.display(A ^ B)

    print("\n---- Transpose\n")
    af.display(A)
    af.display(af.transpose(A))

    print("\n---- Flip Vertically / Horizontally\n")
    af.display(A)
    af.display(af.flip(A, 0))
    af.display(af.flip(A, 1))

    print("\n---- Sum, Min, Max along row / columns\n")
    af.display(A)
    af.display(af.sum(A, 0))
    af.display(af.min(A, 0))
    af.display(af.max(A, 0))
    af.display(af.sum(A, 1))
    af.display(af.min(A, 1))
    af.display(af.max(A, 1))

    print("\n---- Get minimum with index\n")
    (min_val, min_idx) = af.imin(A, 0)
    af.display(min_val)
    af.display(min_idx)
