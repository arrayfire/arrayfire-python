#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from __future__ import absolute_import

import sys
from .simple_tests import *

tests = {}
tests['simple'] = simple.tests

def assert_valid(name, name_list, name_str):
    is_valid = any([name == val for val in name_list])
    if not is_valid:
        err_str  = "The first argument needs to be a %s name\n" % name_str
        err_str += "List of supported %ss: %s" % (name_str, str(list(name_list)))
        raise RuntimeError(err_str)

if __name__ == "__main__":

    module_name = None
    num_args = len(sys.argv)

    if (num_args > 1):
        module_name = sys.argv[1].lower()
        assert_valid(sys.argv[1].lower(), tests.keys(), "module")

    if (module_name is None):
        for name in tests:
            tests[name].run()
    else:
        test = tests[module_name]
        test_list = None

        if (num_args > 2):
            test_list = sys.argv[2:]
            for test_name in test_list:
                assert_valid(test_name.lower(), test.keys(), "test")

        test.run(test_list)
