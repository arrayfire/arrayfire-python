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
from .bcast import *
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
        if bcast_var.get() is True:
            bcast_var.toggle()
            raise StopIteration
        else:
            bcast_var.toggle()
            return self

    def __next__(self):
        """
        Function called by the iterator in Python 3
        """
        return self.next()

class _uidx(ct.Union):
    _fields_ = [("arr", ct.c_void_p),
                ("seq", Seq)]

class Index(ct.Structure):
    _fields_ = [("idx", _uidx),
                ("isSeq", ct.c_bool),
                ("isBatch", ct.c_bool)]

    def __init__ (self, idx):

        self.idx     = _uidx()
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
