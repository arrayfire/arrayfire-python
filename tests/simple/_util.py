#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af

def display_func(verbose):
    if (verbose):
        return af.display
    else:
        def eval_func(foo):
            res = foo
        return eval_func

def print_func(verbose):
    def print_func_impl(*args):
        if (verbose):
            print(args)
        else:
            res = [args]
    return print_func_impl

class _simple_test_dict(dict):

    def __init__(self):
        self.print_str = "Simple %16s: %s"
        super(_simple_test_dict, self).__init__()

    def run(self, name_list=None, verbose=False):
        test_list = name_list if name_list is not None else self.keys()
        for key in test_list:

            try:
                test = self[key]
            except:
                print(self.print_str % (key, "NOTFOUND"))
                continue

            try:
                test(verbose)
                print(self.print_str % (key, "PASSED"))
            except:
                print(self.print_str % (key, "FAILED"))

tests = _simple_test_dict()
