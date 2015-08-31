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
from .broadcast import *
import math

class Seq(ct.Structure):
    _fields_ = [("begin", ct.c_double),
                ("end"  , ct.c_double),
                ("step" , ct.c_double)]

    def __init__ (self, S):
        self.begin = ct.c_double( 0)
        self.end   = ct.c_double(-1)
        self.step  = ct.c_double( 1)

        if is_number(S):
            self.begin = ct.c_double(S)
            self.end   = ct.c_double(S)
        elif isinstance(S, slice):
            if (S.step is not None):
                self.step  = ct.c_double(S.step)
                if(S.step < 0):
                    self.begin, self.end = self.end, self.begin
            if (S.start is not None):
                self.begin = ct.c_double(S.start)
            if (S.stop is not None):
                self.end = ct.c_double(S.stop - math.copysign(1, self.step))
        else:
            raise IndexError("Invalid type while indexing arrayfire.array")

class ParallelRange(Seq):

    def __init__(self, start, stop=None, step=None):

        if (stop is None):
            stop = start
            start = 0

        self.S = slice(start, stop, step)
        super(ParallelRange, self).__init__(self.S)

    def __iter__(self):
        return self

    def next(self):
        if bcast.get() is True:
            bcast.toggle()
            raise StopIteration
        else:
            bcast.toggle()
            return self

    def __next__(self):
        return self.next()

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
    _fields_ = [("arr", ct.c_void_p),
                ("seq", Seq)]

class Index(ct.Structure):
    _fields_ = [("idx", uidx),
                ("isSeq", ct.c_bool),
                ("isBatch", ct.c_bool)]

    def __init__ (self, idx):

        self.idx     = uidx()
        self.isBatch = False
        self.isSeq   = True

        if isinstance(idx, BaseArray):
            self.idx.arr = idx.arr
            self.isSeq   = False
        elif isinstance(idx, ParallelRange):
            self.idx.seq = idx
            self.isBatch = True
        else:
            self.idx.seq = Seq(idx)

def get_indices(key):

    index_vec = Index * 4
    S = Index(slice(None))
    inds = index_vec(S, S, S, S)

    if isinstance(key, tuple):
        n_idx = len(key)
        for n in range(n_idx):
            inds[n] = Index(key[n])
    else:
        inds[0] = Index(key)

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
    elif isinstance(key, ParallelRange):
        dims[0] = slice_to_length(key.S, idims[0])
        return dims
    elif isinstance(key, BaseArray):
        dims[0] = key.elements()
        return dims
    elif isinstance(key, tuple):
        n_inds = len(key)

        for n in range(n_inds):
            if (is_number(key[n])):
                dims[n] = 1
            elif (isinstance(key[n], BaseArray)):
                dims[n] = key[n].elements()
            elif (isinstance(key[n], slice)):
                dims[n] = slice_to_length(key[n], idims[n])
            elif (isinstance(key[n], ParallelRange)):
                dims[n] = slice_to_length(key[n].S, idims[n])
            else:
                raise IndexError("Invalid type while assigning to arrayfire.array")

        return dims
    else:
        raise IndexError("Invalid type while assigning to arrayfire.array")
