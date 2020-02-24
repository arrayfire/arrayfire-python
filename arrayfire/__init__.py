#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
ArrayFire is a high performance scientific computing library with an easy to use API.


    >>> # Monte Carlo estimation of pi
    >>> def calc_pi_device(samples):
            # Simple, array based API
            # Generate uniformly distributed random numers
            x = af.randu(samples)
            y = af.randu(samples)
            # Supports Just In Time Compilation
            # The following line generates a single kernel
            within_unit_circle = (x * x + y * y) < 1
            # Intuitive function names
            return 4 * af.count(within_unit_circle) / samples

Programs written using ArrayFire are portable across CUDA, OpenCL and CPU devices.

The default backend is chosen in the following order of preference based on the available libraries:

    1. CUDA
    2. OpenCL
    3. CPU

The backend can be chosen at the beginning of the program by using the following function

    >>> af.set_backend(name)

where name is one of 'cuda', 'opencl' or 'cpu'.

The functionality provided by ArrayFire spans the following domains:

    1. Vector Algorithms
    2. Image Processing
    3. Signal Processing
    4. Computer Vision
    5. Linear Algebra
    6. Statistics

"""

try:
    import pycuda.autoinit
except ImportError:
    pass

from .library    import *
from .array      import *
from .data       import *
from .util       import *
from .algorithm  import *
from .device     import *
from .blas       import *
from .arith      import *
from .statistics import *
from .lapack     import *
from .signal     import *
from .image      import *
from .features   import *
from .vision     import *
from .graphics   import *
from .bcast      import *
from .index      import *
from .interop    import *
from .timer      import *
from .random     import *
from .sparse     import *
from .ml         import *

# do not export default modules as part of arrayfire
del ct
del inspect
del numbers
del os

if (AF_NUMPY_FOUND):
    del np
