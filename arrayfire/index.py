#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
"""
Index and Seq classes used in indexing operations.
"""

from .library import *
from .util import *
from .util import _is_number
from .base import *
from .bcast import _bcast_var
import math

class Seq(ct.Structure):
    """
    arrayfire equivalent of slice

    Attributes
    ----------

    begin: number
           Start of the sequence.

    end  : number
           End of sequence.

    step : number
           Step size.

    Parameters
    ----------

    S: slice or number.

    """
    _fields_ = [("begin", c_double_t),
                ("end"  , c_double_t),
                ("step" , c_double_t)]

    def __init__ (self, S):
        self.begin = c_double_t( 0)
        self.end   = c_double_t(-1)
        self.step  = c_double_t( 1)

        if _is_number(S):
            self.begin = c_double_t(S)
            self.end   = c_double_t(S)
        elif isinstance(S, slice):
            if (S.step is not None):
                self.step  = c_double_t(S.step)
                if(S.step < 0):
                    self.begin, self.end = self.end, self.begin
            if (S.start is not None):
                self.begin = c_double_t(S.start)
            if (S.stop is not None):
                self.end = c_double_t(S.stop)

            # handle special cases
            if self.begin >= 0 and self.end >=0 and self.end <= self.begin and self.step >= 0:
                self.begin = 1
                self.end   = 1
                self.step  = 1
            elif self.begin < 0 and self.end < 0 and self.end >= self.begin and self.step <= 0:
                self.begin = -2
                self.end   = -2
                self.step  = -1

            if (S.stop is not None):
                self.end = self.end - math.copysign(1, self.step)
        else:
            raise IndexError("Invalid type while indexing arrayfire.array")

class ParallelRange(Seq):

    """
    Class used to parallelize for loop.

    Inherits from Seq.

    Attributes
    ----------

    S: slice

    Parameters
    ----------

    start: number
           Beginning of parallel range.

    stop : number
           End of parallel range.

    step : number
           Step size for parallel range.

    Examples
    --------

    >>> import arrayfire as af
    >>> a = af.randu(3, 3)
    >>> b = af.randu(3, 1)
    >>> c = af.constant(0, 3, 3)
    >>> for ii in af.ParallelRange(3):
    ...     c[:, ii] = a[:, ii] + b
    ...
    >>> af.display(a)
    [3 3 1 1]
        0.4107     0.1794     0.3775
        0.8224     0.4198     0.3027
        0.9518     0.0081     0.6456

    >>> af.display(b)
    [3 1 1 1]
        0.7269
        0.7104
        0.5201

    >>> af.display(c)
    [3 3 1 1]
        1.1377     0.9063     1.1045
        1.5328     1.1302     1.0131
        1.4719     0.5282     1.1657

    """
    def __init__(self, start, stop=None, step=None):

        if (stop is None):
            stop = start
            start = 0

        self.S = slice(start, stop, step)
        super(ParallelRange, self).__init__(self.S)

    def __iter__(self):
        return self

    def next(self):
        """
        Function called by the iterator in Python 2
        """
        if _bcast_var.get() is True:
            _bcast_var.toggle()
            raise StopIteration
        else:
            _bcast_var.toggle()
            return self

    def __next__(self):
        """
        Function called by the iterator in Python 3
        """
        return self.next()

class _uidx(ct.Union):
    _fields_ = [("arr", c_void_ptr_t),
                ("seq", Seq)]

class Index(ct.Structure):
    _fields_ = [("idx", _uidx),
                ("isSeq", c_bool_t),
                ("isBatch", c_bool_t)]

    """
    Container for the index class in arrayfire C library

    Attributes
    ----------
    idx.arr: ctypes.c_void_p
             - Default 0

    idx.seq: af.Seq
             - Default af.Seq(0, -1, 1)

    isSeq   : bool
            - Default True

    isBatch : bool
            - Default False

    Parameters
    -----------

    idx: key
         - If of type af.Array, self.idx.arr = idx, self.isSeq = False
         - If of type af.ParallelRange, self.idx.seq = idx, self.isBatch = True
         - Default:, self.idx.seq = af.Seq(idx)

    Note
    ----

    Implemented for internal use only. Use with extreme caution.

    """

    def __init__ (self, idx):

        self.idx     = _uidx()
        self.isBatch = False
        self.isSeq   = True

        if isinstance(idx, BaseArray):

            arr = c_void_ptr_t(0)

            if (idx.type() == Dtype.b8.value):
                safe_call(backend.get().af_where(c_pointer(arr), idx.arr))
            else:
                safe_call(backend.get().af_retain_array(c_pointer(arr), idx.arr))

            self.idx.arr = arr
            self.isSeq   = False
        elif isinstance(idx, ParallelRange):
            self.idx.seq = idx
            self.isBatch = True
        else:
            self.idx.seq = Seq(idx)

    def __del__(self):
        if not self.isSeq:
            # ctypes field variables are automatically
            # converted to basic C types so we have to
            # build the void_p from the value again.
            arr = c_void_ptr_t(self.idx.arr)
            backend.get().af_release_array(arr)

_span = Index(slice(None))
class _Index4(object):
    def __init__(self):
        index_vec = Index * 4
        self.array = index_vec(_span, _span, _span, _span)
        # Do not lose those idx as self.array keeps
        # no reference to them. Otherwise the destructor
        # is prematurely called
        self.idxs = [_span, _span, _span, _span]
    @property
    def pointer(self):
        return c_pointer(self.array)

    def __getitem__(self, idx):
        return self.array[idx]

    def __setitem__(self, idx, value):
        self.array[idx] = value
        self.idxs[idx] = value
