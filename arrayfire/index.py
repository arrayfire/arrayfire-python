#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
from .library import *
from .util import *
from .base import *

class seq(ct.Structure):
    _fields_ = [("begin", ct.c_double),
                ("end"  , ct.c_double),
                ("step" , ct.c_double)]

    def __init__ (self, S):
        num = __import__("numbers")

        self.begin = ct.c_double( 0)
        self.end   = ct.c_double(-1)
        self.step  = ct.c_double( 1)

        if is_number(S):
            self.begin = ct.c_double(S)
            self.end   = ct.c_double(S)
        elif isinstance(S, slice):
            if (S.start is not None):
                self.begin = ct.c_double(S.start)
            if (S.stop is not None):
                self.end   = ct.c_double(S.stop - 1)
            if (S.step is not None):
                self.step  = ct.c_double(S.step)
        else:
            raise IndexError("Invalid type while indexing arrayfire.array")

def slice_to_length(key, dim):
    tkey = [key.start, key.stop, key.step]

    if tkey[0] is None:
        tkey[0] = 0
    elif tkey[0] < 0:
        tkey[0] = dim - tkey[0]

    if tkey[1] is None:
        tkey[1] = dim
    elif tkey[1] < 0:
        tkey[1] = dim - tkey[1]

    if tkey[2] is None:
        tkey[2] = 1

    return int(((tkey[1] - tkey[0] - 1) / tkey[2]) + 1)

class uidx(ct.Union):
    _fields_ = [("arr", ct.c_longlong),
                ("seq", seq)]

class index(ct.Structure):
    _fields_ = [("idx", uidx),
                ("isSeq", ct.c_bool),
                ("isBatch", ct.c_bool)]

    def __init__ (self, idx):

        self.idx     = uidx()
        self.isBatch = False
        self.isSeq   = True

        if isinstance(idx, base_array):
            self.idx.arr = idx.arr
            self.isSeq   = False
        else:
            self.idx.seq = seq(idx)

def get_indices(key, n_dims):

    index_vec = index * n_dims
    inds = index_vec()

    for n in range(n_dims):
        inds[n] = index(slice(None))

    if isinstance(key, tuple):
        n_idx = len(key)
        for n in range(n_idx):
            inds[n] = index(key[n])
    else:
        inds[0] = index(key)

    return inds

def get_assign_dims(key, idims):

    dims = [1]*4

    for n in range(len(idims)):
        dims[n] = idims[n]

    if is_number(key):
        dims[0] = 1
        return dims
    elif isinstance(key, slice):
        dims[0] = slice_to_length(key, idims[0])
        return dims
    elif isinstance(key, base_array):
        dims[0] = key.elements()
        return dims
    elif isinstance(key, tuple):
        n_inds = len(key)

        if (n_inds > len(idims)):
            raise IndexError("Number of indices greater than array dimensions")

        for n in range(n_inds):
            if (is_number(key[n])):
                dims[n] = 1
            elif (isinstance(key[n], base_array)):
                dims[n] = key[n].elements()
            elif (isinstance(key[n], slice)):
                dims[n] = slice_to_length(key[n], idims[n])
            else:
                raise IndexError("Invalid type while assigning to arrayfire.array")

        return dims
    else:
        raise IndexError("Invalid type while assigning to arrayfire.array")
