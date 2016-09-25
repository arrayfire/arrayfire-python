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
            self.engine  = c_void_ptr_t(0)
            safe_call(backend.get().af_create_random_engine(c_pointer(self.engine), engine_type.value, c_longlong_t(seed)))
        else:
            self.engine = engine

    def __del__(self):
        safe_call(backend.get().af_release_random_engine(self.engine))

    def set_type(self, engine_type):
        """
        Set the type of the random engine.
        """
        safe_call(backend.get().af_random_engine_set_type(c_pointer(self.engine), engine_type.value))

    def get_type(self):
        """
        Get the type of the random engine.
        """
        __to_random_engine_type = [RANDOM_ENGINE.PHILOX_4X32_10,
                                   RANDOM_ENGINE.THREEFRY_2X32_16,
                                   RANDOM_ENGINE.MERSENNE_GP11213]
        rty = c_int_t(RANDOM_ENGINE.PHILOX.value)
        safe_call(backend.get().af_random_engine_get_type(c_pointer(rty), self.engine))
        return __to_random_engine_type[rty]

    def set_seed(self, seed):
        """
        Set the seed for the random engine.
        """
        safe_call(backend.get().af_random_engine_set_seed(c_pointer(self.engine), c_longlong_t(seed)))

    def get_seed(self):
        """
        Get the seed for the random engine.
        """
        seed = c_longlong_t(0)
        safe_call(backend.get().af_random_engine_get_seed(c_pointer(seed), self.engine))
        return seed.value

def randu(d0, d1=None, d2=None, d3=None, dtype=Dtype.f32, engine=None):
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

    engine : optional: Random_Engine. default: None.
             If engine is None, uses a default engine created by arrayfire.

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

    if engine is None:
        safe_call(backend.get().af_randu(c_pointer(out.arr), 4, c_pointer(dims), dtype.value))
    else:
        safe_call(backend.get().af_random_uniform(c_pointer(out.arr), 4, c_pointer(dims), dtype.value, engine.engine))

    return out

def randn(d0, d1=None, d2=None, d3=None, dtype=Dtype.f32, engine=None):
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

    engine : optional: Random_Engine. default: None.
             If engine is None, uses a default engine created by arrayfire.

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

    if engine is None:
        safe_call(backend.get().af_randn(c_pointer(out.arr), 4, c_pointer(dims), dtype.value))
    else:
        safe_call(backend.get().af_random_normal(c_pointer(out.arr), 4, c_pointer(dims), dtype.value, engine.engine))

    return out

def set_seed(seed=0):
    """
    Set the seed for the random number generator.

    Parameters
    ----------
    seed: int.
          Seed for the random number generator
    """
    safe_call(backend.get().af_set_seed(c_ulonglong_t(seed)))

def get_seed():
    """
    Get the seed for the random number generator.

    Returns
    -------
    seed: int.
          Seed for the random number generator
    """
    seed = c_ulonglong_t(0)
    safe_call(backend.get().af_get_seed(c_pointer(seed)))
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
    safe_call(backend.get().af_set_default_random_engine_type(c_pointer(self.engine), engine_type.value))

def get_default_random_engine():
    """
    Get the default random engine

    Returns
    -------

    The default random engine used by randu and randn
    """
    engine = c_void_ptr_t(0)
    default_engine = c_void_ptr_t(0)
    safe_call(backend.get().af_get_default_random_engine(c_pointer(default_engine)))
    safe_call(backend.get().af_retain_random_engine(c_pointer(engine), default_engine))
    return Random_Engine(engine=engine)
