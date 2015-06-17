from .library import *

class array(object):

    def __init__(self):
        self.arr = c_longlong(0)

    def __del__(self):
        if (self.arr.value != 0):
            clib.af_release_array(self.arr)
