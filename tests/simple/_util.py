#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################
import traceback
import logging
import arrayfire as af
import sys

class _simple_test_dict(dict):

    def __init__(self):
        self.print_str = "Simple %16s: %s"
        self.failed = False
        super(_simple_test_dict, self).__init__()

    def run(self, name_list=None, verbose=False):
        test_list = name_list if name_list is not None else self.keys()
        for key in test_list:
            self.print_log = ''
            try:
                test = self[key]
            except:
                print(self.print_str % (key, "NOTFOUND"))
                continue

            try:
                test(verbose)
                print(self.print_str % (key, "PASSED"))
            except Exception as e:
                print(self.print_str % (key, "FAILED"))
                self.failed = True
                if (not verbose):
                    print(tests.print_log)
                logging.error(traceback.format_exc())

        if (self.failed):
            sys.exit(1)

tests = _simple_test_dict()

def print_func(verbose):
    def print_func_impl(*args):
        _print_log = ''
        for arg in args:
            _print_log += str(arg) + '\n'
        if (verbose):
            print(_print_log)
        tests.print_log += _print_log
    return print_func_impl

def display_func(verbose):
    return print_func(verbose)
