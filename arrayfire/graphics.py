#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
graphics functions for arrayfire
"""

from .library import *
from .array import *
from .util import _is_number

class _Cell(ct.Structure):
    _fields_ = [("row", ct.c_int),
                ("col", ct.c_int),
                ("title", ct.c_char_p),
                ("cmap", ct.c_int)]

    def __init__(self, r, c, title, cmap):
        self.row = r
        self.col = c
        self.title = title if title is not None else ct.c_char_p()
        self.cmap = cmap.value

class Window(object):
    """
    Class to create the Window object.

    Parameters
    ----------

    width: optional: int. default: 1280.
           - Specifies the width of the window in pixels.

    height: optional: int. default: 720.
           - Specifies the height of the window in pixels.

    title: optional: str. default: "ArrayFire".
          - Specifies the title used for the window.

    """

    def __init__(self, width=1280, height=720, title="ArrayFire"):
        self._r = -1
        self._c = -1
        self._wnd = ct.c_void_p(0)
        self._cmap = COLORMAP.DEFAULT

        _width  = 1280 if  width is None else  width
        _height =  720 if height is None else height
        _title  = "ArrayFire" if title is None else title

        _title = _title.encode("ascii")

        safe_call(backend.get().af_create_window(ct.pointer(self._wnd),
                                                 ct.c_int(_width), ct.c_int(_height),
                                                 ct.c_char_p(_title)))

    def __del__(self):
        """
        Destroys the window when going out of scope.
        """
        safe_call(backend.get().af_destroy_window(self._wnd))

    def set_pos(self, x, y):
        """
        Set the position of window on the screen.

        Parameters
        ----------

        x : int.
            Pixel offset from left.

        y : int.
            Pixel offset from top

        """
        safe_call(backend.get().af_set_position(self._wnd, ct.c_int(x), ct.c_int(y)))

    def set_title(self, title):
        """
        Set the title of the window

        Parameters
        ----------

        title : str.
            Title used for the current window.

        """
        safe_call(backend.get().af_set_title(self._wnd, title))

    def set_colormap(self, cmap):
        """
        Set the colormap for the window.

        Parameters
        ----------

        cmap : af.COLORMAP.
            Set the colormap for the window.

        """
        self._cmap = cmap

    def set_size(self, w, h):
        """
        Set the windo height and width.

        Parameters
        -----------
        w  : int
           Width if window.

        h  : int
           Height of window.
        """
        safe_call(backend.get().af_set_size(self._wnd, w, h))

    def image(self, img, title=None):
        """
        Display an arrayfire array as an image.

        Paramters
        ---------

        img: af.Array.
             A 2 dimensional array for single channel image.
             A 3 dimensional array for 3 channel image.

        title: str.
             Title used for the image.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_image(self._wnd, img.arr, ct.pointer(_cell)))

    def scatter(self, X, Y, marker=MARKER.POINT, title=None):
        """
        Renders input arrays as 2D scatter plot.

        Paramters
        ---------

        X: af.Array.
             A 1 dimensional array containing X co-ordinates.

        Y: af.Array.
             A 1 dimensional array containing Y co-ordinates.

        marker: af.MARKER
             Specifies how the points look

        title: str.
             Title used for the plot.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_scatter(self._wnd, X.arr, Y.arr,
                                                marker.value, ct.pointer(_cell)))

    def scatter3(self, P, marker=MARKER.POINT, title=None):
        """
        Renders the input array as a 3D Scatter plot.

        Paramters
        ---------

        P: af.Array.
             A 2 dimensional array containing (X,Y,Z) co-ordinates.

        title: str.
             Title used for the plot.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_scatter3(self._wnd, P.arr,
                                                 marker.value, ct.pointer(_cell)))

    def plot(self, X, Y, title=None):
        """
        Display a 2D Plot.

        Paramters
        ---------

        X: af.Array.
             A 1 dimensional array containing X co-ordinates.

        Y: af.Array.
             A 1 dimensional array containing Y co-ordinates.

        title: str.
             Title used for the plot.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_plot(self._wnd, X.arr, Y.arr, ct.pointer(_cell)))

    def plot3(self, line, title=None):
        """
        Renders the input array as a 3D line plot.

        Paramters
        ---------

        line: af.Array.
             A 2 dimensional array containing (X,Y,Z) co-ordinates.

        title: str.
             Title used for the plot.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_plot3(self._wnd, line.arr, ct.pointer(_cell)))

    def surface(self, x_vals, y_vals, z_vals, title=None):
        """
        Renders the input array as a 3D surface plot.

        Paramters
        ---------

        x_vals: af.Array.
             A 1 dimensional array containing X co-ordinates.

        y_vals: af.Array.
             A 1 dimensional array containing Y co-ordinates.

        z_vals: af.Array.
             A 1 dimensional array containing Z co-ordinates.

        title: str.
             Title used for the plot.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_surface(self._wnd,
                                                x_vals.arr, y_vals.arr, z_vals.arr,
                                                ct.pointer(_cell)))

    def hist(self, X, min_val, max_val, title=None):
        """
        Display a histogram Plot.

        Paramters
        ---------

        X: af.Array.
             A 1 dimensional array containing the histogram.

        min_val: scalar.
             A scalar value specifying the lower bound of the histogram.

        max_val: scalar.
             A scalar value specifying the upper bound of the histogram.

        title: str.
             Title used for the histogram.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_hist(self._wnd, X.arr,
                                             ct.c_double(max_val), ct.c_double(min_val),
                                             ct.pointer(_cell)))

    def grid(self, rows, cols):
        """
        Create a grid for sub plotting within the window.

        Parameters
        ----------

        rows: int.
              Number of rows in the grid.

        cols: int.
              Number of columns in the grid.

        """
        safe_call(backend.get().af_grid(self._wnd, ct.c_int(rows), ct.c_int(cols)))

    def show(self):
        """
        Force the window to display the contents.

        Note: This is only needed when using the window as a grid.
        """
        safe_call(backend.get().af_show(self._wnd))

    def close(self):
        """
        Close the window.
        """
        tmp = ct.c_bool(True)
        safe_call(backend.get().af_is_window_closed(ct.pointer(tmp), self._wnd))
        return tmp

    def set_visibility(is_visible):
        """
        A flag that shows or hides the window as requested.

        Parameters
        ----------
        is_visible: Flag specifying the visibility of the flag.
        """
        safe_call(backend.get().af_set_visibility(self._wnd, is_visible))

    def __getitem__(self, keys):
        """
        Get access to a specific grid location within the window.

        Examples
        --------

        >>> a = af.randu(5,5)
        >>> b = af.randu(5,5)
        >>> w = af.Window()
        >>> w.grid(1,2)
        >>> w[0, 0].image(a)
        >>> w[0, 1].image(b)
        >>> w.show()
        """
        if not isinstance(keys, tuple):
            raise IndexError("Window expects indexing along two dimensions")
        if len(keys) != 2:
            raise IndexError("Window expects indexing along two dimensions only")
        if not (_is_number(keys[0]) and _is_number(keys[1])):
            raise IndexError("Window expects the indices to be numbers")
        self._r = keys[0]
        self._c = keys[1]

        return self
