#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################


class bcast(object):
    _flag = False
    def get():
        return bcast._flag

    def set(flag):
        bcast._flag = flag

    def toggle():
        bcast._flag ^= True

def broadcast(func, *args):
    bcast.toggle()
    res = func(*args)
    bcast.toggle()
    return res
