#!/usr/bin/env python

#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

def reverse_char(b):
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1
    return b


# http://stackoverflow.com/a/9144870/2192361
def reverse(x):
    x = ((x >> 1)  & 0x55555555) | ((x & 0x55555555) << 1)
    x = ((x >> 2)  & 0x33333333) | ((x & 0x33333333) << 2)
    x = ((x >> 4)  & 0x0f0f0f0f) | ((x & 0x0f0f0f0f) << 4)
    x = ((x >> 8)  & 0x00ff00ff) | ((x & 0x00ff00ff) << 8)
    x = ((x >> 16) & 0xffff)     | ((x & 0xffff)     << 16);
    return x


def read_idx(name):
    with open(name, 'rb') as f:
        # In the C++ version, bytes the size of 4 chars are being read
        # May not work properly in machines where a char is not 1 byte
        bytes_read = f.read(4)
        bytes_read = bytearray(bytes_read)

        if bytes_read[2] != 8:
            raise RuntimeError('Unsupported data type')

        numdims = bytes_read[3]
        elemsize = 1

        # Read the dimensions
        elem = 1
        dims = [0] * numdims
        for i in range(numdims):
            bytes_read = bytearray(f.read(4))

            # Big endian to little endian
            for j in range(4):
                bytes_read[j] = reverse_char(bytes_read[j])
            bytes_read_int = int.from_bytes(bytes_read, 'little')
            dim = reverse(bytes_read_int)

            elem = elem * dim;
            dims[i] = dim;

        # Read the data
        cdata = f.read(elem * elemsize)
        cdata = list(cdata)
        data = [float(cdata_elem) for cdata_elem in cdata]

        return (dims, data)


