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

try:
    # Display backend information
    af.info()

    print("Create a 5-by-3 matrix of random floats on the GPU\n")
    A = af.randu(5, 3, 1, 1, af.Dtype.f32)
    af.display(A)

    print("Element-wise arithmetic\n")
    B = af.sin(A) + 1.5
    af.display(B)

    print("Negate the first three elements of second column\n")
    B[0:3, 1] = B[0:3, 1] * -1
    af.display(B)

    print("Fourier transform the result\n");
    C = af.fft(B);
    af.display(C);

    print("Grab last row\n");
    c = C[-1,:];
    af.display(c);

    print("Scan Test\n");
    r = af.constant(2, 16, 4, 1, 1);
    af.display(r);

    print("Scan\n");
    S = af.scan(r, 0, af.BINARYOP.MUL);
    af.display(S);

    print("Create 2-by-3 matrix from host data\n");
    d = [ 1, 2, 3, 4, 5, 6 ]
    D = af.Array(d, (2, 3))
    af.display(D)

    print("Copy last column onto first\n");
    D[:,0] = D[:, -1]
    af.display(D);

    print("Sort A and print sorted array and corresponding indices\n");
    [sorted_vals, sorted_idxs] = af.sort_index(A);
    af.display(A)
    af.display(sorted_vals)
    af.display(sorted_idxs)
except Exception as e:
    print("Error: " + str(e))
