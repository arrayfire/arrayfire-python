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

a = af.randu(5, 5)
b = af.randu(5, 5)
w = af.randu(5, 1)

af.display(af.mean(a, dim=0))
af.display(af.mean(a, weights=w, dim=0))
print(af.mean(a))
print(af.mean(a, weights=w))

af.display(af.var(a, dim=0))
af.display(af.var(a, isbiased=True, dim=0))
af.display(af.var(a, weights=w, dim=0))
print(af.var(a))
print(af.var(a, isbiased=True))
print(af.var(a, weights=w))

af.display(af.stdev(a, dim=0))
print(af.stdev(a))

af.display(af.var(a, dim=0))
af.display(af.var(a, isbiased=True, dim=0))
print(af.var(a))
print(af.var(a, isbiased=True))

af.display(af.median(a, dim=0))
print(af.median(w))

print(af.corrcoef(a, b))
