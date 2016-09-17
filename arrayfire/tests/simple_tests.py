#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from __future__ import absolute_import

from . import simple
import sys

if __name__ == "__main__":
    verbose = False

    if len(sys.argv) > 1:
        verbose = int(sys.argv[1])

    test_list = None
    if len(sys.argv) > 2:
        test_list = sys.argv[2:]

    simple.tests.run(test_list, verbose)
