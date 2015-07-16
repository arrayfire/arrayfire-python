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

a = af.randu(5,5)
b = af.randu(5,5)

af.print_array(af.matmul(a,b))
af.print_array(af.matmul(a,b,af.AF_MAT_TRANS))
af.print_array(af.matmul(a,b,af.AF_MAT_NONE, af.AF_MAT_TRANS))

b = af.randu(5,1)
af.print_array(af.dot(b,b))

af.print_array(af.transpose(a))

af.transpose_inplace(a)
af.print_array(a)
