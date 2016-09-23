#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Random engine class and functions to generate random numbers.
"""

from .library import *
from .array import *
import numbers

class Random_Engine(object):
    """
    Class to handle random number generator engines.

    Parameters
    ----------

    engine_type : optional: RANDOME_ENGINE. default: RANDOM_ENGINE.PHILOX
                - Specifies the type of random engine to be created. Can be one of:
                - RANDOM_ENGINE.PHILOX_4X32_10
                - RANDOM_ENGINE.THREEFRY_2X32_16
                - RANDOM_ENGINE.MERSENNE_GP11213
                - RANDOM_ENGINE.PHILOX (same as RANDOM_ENGINE.PHILOX_4X32_10)
                - RANDOM_ENGINE.THREEFRY (same as RANDOM_ENGINE.THREEFRY_2X32_16)
                - RANDOM_ENGINE.DEFAULT
                - Not used if engine is not None

    seed        : optional int. default: 0
                - Specifies the seed for the random engine
                - Not used if engine is not None

    engine      : optional ctypes.c_void_p. default: None.
                - Used a handle created by the C api to create the Random_Engine.
    """

    def __init__(self, engine_type = RANDOM_ENGINE.PHILOX, seed = 0, engine = None):
        if (engine is None):
            self.engine  = ct.c_void_p(0)
            safe_call(backend.get().af_create_random_engine(ct.pointer(self.engine), engine_type.value, ct.c_longlong(seed)))
        else:
            self.engine = engine

    def __del__(self):
        safe_call(backend.get().af_release_random_engine(self.engine))

    def set_type(self, engine_type):
        """
        Set the type of the random engine.
        """
        safe_call(backend.get().af_random_engine_set_type(ct.pointer(self.engine), engine_type.value))

    def get_type(self):
        """
        Get the type of the random engine.
        """
        __to_random_engine_type = [RANDOM_ENGINE.PHILOX_4X32_10,
                                   RANDOM_ENGINE.THREEFRY_2X32_16,
                                   RANDOM_ENGINE.MERSENNE_GP11213]
        rty = ct.c_int(RANDOM_ENGINE.PHILOX.value)
        safe_call(backend.get().af_random_engine_get_type(ct.pointer(rty), self.engine))
        return __to_random_engine_type[rty]

    def set_seed(self, seed):
        """
        Set the seed for the random engine.
        """
        safe_call(backend.get().af_random_engine_set_seed(ct.pointer(self.engine), ct.c_longlong(seed)))

    def get_seed(self):
        """
        Get the seed for the random engine.
        """
        seed = ct.c_longlong(0)
        safe_call(backend.get().af_random_engine_get_seed(ct.pointer(seed), self.engine))
        return seed.value

def randu(d0, d1=None, d2=None, d3=None, dtype=Dtype.f32, random_engine=None):
    """
    Create a multi dimensional array containing values from a uniform distribution.

    Parameters
    ----------
    d0 : int.
         Length of first dimension.

    d1 : optional: int. default: None.
         Length of second dimension.

    d2 : optional: int. default: None.
         Length of third dimension.

    d3 : optional: int. default: None.
         Length of fourth dimension.

    dtype : optional: af.Dtype. default: af.Dtype.f32.
           Data type of the array.

    random_engine : optional: Random_Engine. default: None.
             If random_engine is None, uses a default engine created by arrayfire.

    Returns
    -------

    out : af.Array
          Multi dimensional array whose elements are sampled uniformly between [0, 1].
          - If d1 is None, `out` is 1D of size (d0,).
          - If d1 is not None and d2 is None, `out` is 2D of size (d0, d1).
          - If d1 and d2 are not None and d3 is None, `out` is 3D of size (d0, d1, d2).
          - If d1, d2, d3 are all not None, `out` is 4D of size (d0, d1, d2, d3).
    """
    out = Array()
    dims = dim4(d0, d1, d2, d3)

    if random_engine is None:
        safe_call(backend.get().af_randu(ct.pointer(out.arr), 4, ct.pointer(dims), dtype.value))
    else:
        safe_call(backend.get().af_random_uniform(ct.pointer(out.arr), 4, ct.pointer(dims), random_engine.engine))

    return out

def randn(d0, d1=None, d2=None, d3=None, dtype=Dtype.f32, random_engine=None):
    """
    Create a multi dimensional array containing values from a normal distribution.

    Parameters
    ----------
    d0 : int.
         Length of first dimension.

    d1 : optional: int. default: None.
         Length of second dimension.

    d2 : optional: int. default: None.
         Length of third dimension.

    d3 : optional: int. default: None.
         Length of fourth dimension.

    dtype : optional: af.Dtype. default: af.Dtype.f32.
           Data type of the array.

    random_engine : optional: Random_Engine. default: None.
             If random_engine is None, uses a default engine created by arrayfire.

    Returns
    -------

    out : af.Array
          Multi dimensional array whose elements are sampled from a normal distribution with mean 0 and sigma of 1.
          - If d1 is None, `out` is 1D of size (d0,).
          - If d1 is not None and d2 is None, `out` is 2D of size (d0, d1).
          - If d1 and d2 are not None and d3 is None, `out` is 3D of size (d0, d1, d2).
          - If d1, d2, d3 are all not None, `out` is 4D of size (d0, d1, d2, d3).
    """

    out = Array()
    dims = dim4(d0, d1, d2, d3)

    if random_engine is None:
        safe_call(backend.get().af_randn(ct.pointer(out.arr), 4, ct.pointer(dims), dtype.value))
    else:
        safe_call(backend.get().af_random_normal(ct.pointer(out.arr), 4, ct.pointer(dims), random_engine.engine))

    return out

def set_seed(seed=0):
    """
    Set the seed for the random number generator.

    Parameters
    ----------
    seed: int.
          Seed for the random number generator
    """
    safe_call(backend.get().af_set_seed(ct.c_ulonglong(seed)))

def get_seed():
    """
    Get the seed for the random number generator.

    Returns
    ----------
    seed: int.
          Seed for the random number generator
    """
    seed = ct.c_ulonglong(0)
    safe_call(backend.get().af_get_seed(ct.pointer(seed)))
    return seed.value

def set_default_random_engine_type(engine_type):
    """
    Set random engine type for default random engine.

    Parameters
    ----------
    engine_type : RANDOME_ENGINE.
                - Specifies the type of random engine to be created. Can be one of:
                - RANDOM_ENGINE.PHILOX_4X32_10
                - RANDOM_ENGINE.THREEFRY_2X32_16
                - RANDOM_ENGINE.MERSENNE_GP11213
                - RANDOM_ENGINE.PHILOX (same as RANDOM_ENGINE.PHILOX_4X32_10)
                - RANDOM_ENGINE.THREEFRY (same as RANDOM_ENGINE.THREEFRY_2X32_16)
                - RANDOM_ENGINE.DEFAULT

    Note
    ----

    This only affects randu and randn when a random engine is not specified.
    """
    safe_call(backend.get().af_set_default_random_engine_type(ct.pointer(self.engine), engine_type.value))

def get_default_random_engine():
    """
    Get the default random engine

    Returns
    ------

    The default random engine used by randu and randn
    """
    engine = ct.c_void_p(0)
    default_engine = ct.c_void_p(0)
    safe_call(backend.get().af_get_default_random_engine(ct.pointer(default_engine)))
    safe_call(backend.get().af_retain_random_engine(ct.pointer(engine), default_engine))
    return Random_Engine(engine=engine)
