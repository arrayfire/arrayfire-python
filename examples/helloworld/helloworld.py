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

# Display backend information
af.info()

# Generate a uniform random array with a size of 5 elements
a = af.randu(5, 1)

# Print a and its minimum value
print(a)

# Print min and max values of a
print("Minimum, Maximum: ", af.min(a), af.max(a))
