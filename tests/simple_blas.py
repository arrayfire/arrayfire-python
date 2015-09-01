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

af.display(af.matmul(a,b))
af.display(af.matmul(a,b,af.MATPROP.TRANS))
af.display(af.matmul(a,b,af.MATPROP.NONE, af.MATPROP.TRANS))

b = af.randu(5,1)
af.display(af.dot(b,b))
