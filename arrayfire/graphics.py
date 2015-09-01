#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from .library import *
from .array import *

class _Cell(ct.Structure):
    _fields_ = [("row", ct.c_int),
                ("col", ct.c_int),
                ("title", ct.c_char_p),
                ("cmap", ct.c_int)]

    def __init__(self, r, c, title, cmap):
        self.row = r
        self.col = c
        self.title = title if title is not None else ct.c_char_p()
        self.cmap = cmap

class window(object):

    def __init__(self, width=None, height=None, title=None):
        self._r = -1
        self._c = -1
        self._wnd = ct.c_longlong(0)
        self._cmap = AF_COLORMAP_DEFAULT

        _width  = 1280 if  width is None else  width
        _height =  720 if height is None else height
        _title  = "ArrayFire" if title is None else title

        _title = _title.encode("ascii")

        safe_call(backend.get().af_create_window(ct.pointer(self._wnd),
                                                 ct.c_int(_width), ct.c_int(_height), ct.c_char_p(_title)))

    def __del__(self):
        safe_call(backend.get().af_destroy_window(self._wnd))

    def set_pos(self, x, y):
        safe_call(backend.get().af_set_position(self._wnd, ct.c_int(x), ct.c_int(y)))

    def set_title(self, title):
        safe_call(backend.get().af_set_title(self._wnd, title))

    def set_colormap(self, cmap):
        self._cmap = cmap

    def image(self, img, title=None):
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_image(self._wnd, img.arr, ct.pointer(_cell)))

    def plot(self, X, Y, title=None):
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_plot(self._wnd, X.arr, Y.arr, ct.pointer(_cell)))

    def hist(self, X, min_val, max_val, title=None):
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_hist(self._wnd, X.arr,
                                             ct.c_double(max_val), ct.c_double(min_val),
                                             ct.pointer(_cell)))

    def grid(rows, cols):
        safe_call(af_grid(self._wnd, ct.c_int(rows), ct.c_int(cols)))

    def show(self):
        safe_call(backend.get().af_show(self._wnd))

    def close(self):
        tmp = ct.c_bool(True)
        safe_call(backend.get().af_is_window_closed(ct.pointer(tmp), self._wnd))
        return tmp

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            raise IndexError("Window expects indexing along two dimensions")
        if len(keys) != 2:
            raise IndexError("Window expects indexing along two dimensions only")
        if not (is_number(keys[0]) and is_number(keys[1])):
            raise IndexError("Window expects the indices to be numbers")
        self._r = keys[0]
        self._c = keys[1]
