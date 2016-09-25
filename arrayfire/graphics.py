#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Graphics functions (plot, image, etc).
"""

from .library import *
from .array import *
from .util import _is_number

class _Cell(ct.Structure):
    _fields_ = [("row", c_int_t),
                ("col", c_int_t),
                ("title", c_char_ptr_t),
                ("cmap", c_int_t)]

    def __init__(self, r, c, title, cmap):
        self.row = r
        self.col = c
        self.title = title if title is not None else c_char_ptr_t()
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
        self._wnd = c_void_ptr_t(0)
        self._cmap = COLORMAP.DEFAULT

        _width  = 1280 if  width is None else  width
        _height =  720 if height is None else height
        _title  = "ArrayFire" if title is None else title

        _title = _title.encode("ascii")

        safe_call(backend.get().af_create_window(c_pointer(self._wnd),
                                                 c_int_t(_width), c_int_t(_height),
                                                 c_char_ptr_t(_title)))

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
        safe_call(backend.get().af_set_position(self._wnd, c_int_t(x), c_int_t(y)))

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

        Parameters
        ----------

        img: af.Array.
             A 2 dimensional array for single channel image.
             A 3 dimensional array for 3 channel image.

        title: str.
             Title used for the image.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_image(self._wnd, img.arr, c_pointer(_cell)))

    def scatter(self, X, Y, Z=None, points=None, marker=MARKER.POINT, title=None):
        """
        Renders input arrays as 2D or 3D scatter plot.

        Parameters
        ----------

        X: af.Array.
             A 1 dimensional array containing X co-ordinates.

        Y: af.Array.
             A 1 dimensional array containing Y co-ordinates.

        Z: optional: af.Array. default: None.
             - A 1 dimensional array containing Z co-ordinates.
             - Not used if line is not None

        points: optional: af.Array. default: None.
             - A 2 dimensional array of size [n 2]. Each column denotes X and Y co-ordinates for 2D scatter plot.
             - A 3 dimensional array of size [n 3]. Each column denotes X, Y, and Z co-ordinates for 3D scatter plot.

        marker: af.MARKER
             Specifies how the points look

        title: str.
             Title used for the plot.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)

        if points is None:
            if Z is None:
                safe_call(backend.get().af_draw_scatter_2d(self._wnd, X.arr, Y.arr,
                                                           marker.value, c_pointer(_cell)))
            else:
                safe_call(backend.get().af_draw_scatter_3d(self._wnd, X.arr, Y.arr, Z.arr,
                                                           marker.value, c_pointer(_cell)))
        else:
            safe_call(backend.get().af_draw_scatter_nd(self._wnd, points.arr, marker.value, c_pointer(_cell)))

    def scatter2(self, points, marker=MARKER.POINT, title=None):
        """
        Renders the input array as a 2D Scatter plot.

        Parameters
        ----------

        points: af.Array.
             A 2 dimensional array containing (X,Y) co-ordinates.

        marker: af.MARKER
             Specifies how the points look

        title: str.
             Title used for the plot.
        """
        assert(points.numdims() == 2)
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_scatter2(self._wnd, points.arr,
                                                 marker.value, c_pointer(_cell)))

    def scatter3(self, points, marker=MARKER.POINT, title=None):
        """
        Renders the input array as a 3D Scatter plot.

        Parameters
        ----------

        points: af.Array.
             A 2 dimensional array containing (X,Y,Z) co-ordinates.

        marker: af.MARKER
             Specifies how the points look

        title: str.
             Title used for the plot.
        """
        assert(points.numdims() == 3)
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_scatter3(self._wnd, points.arr,
                                                 marker.value, c_pointer(_cell)))
    def plot(self, X, Y, Z=None, line = None, title=None):
        """
        Display a 2D or 3D Plot.

        Parameters
        ----------

        X: af.Array.
             - A 1 dimensional array containing X co-ordinates.
             - Not used if line is not None

        Y: af.Array.
             - A 1 dimensional array containing Y co-ordinates.
             - Not used if line is not None

        Z: optional: af.Array. default: None.
             - A 1 dimensional array containing Z co-ordinates.
             - Not used if line is not None

        line: optional: af.Array. default: None.
             - A 2 dimensional array of size [n 2]. Each column denotes X and Y co-ordinates for plotting 2D lines.
             - A 3 dimensional array of size [n 3]. Each column denotes X, Y, and Z co-ordinates for plotting 3D lines.

        title: str.
             Title used for the plot.

        Note
        ----

        The line parameter takes precedence.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        if line is None:
            if Z is None:
                safe_call(backend.get().af_draw_plot_2d(self._wnd, X.arr, Y.arr, c_pointer(_cell)))
            else:
                safe_call(backend.get().af_draw_plot_3d(self._wnd, X.arr, Y.arr, Z.arr, c_pointer(_cell)))
        else:
            safe_call(backend.get().af_draw_plot_nd(self._wnd, line.arr, c_pointer(_cell)))

    def plot2(self, line, title=None):
        """
        Display a 2D Plot.

        Parameters
        ----------

        line: af.Array.
             - A 2 dimensional array of size [n 2]. Each column denotes X, and Y co-ordinates for plotting 2D lines.

        title: str.
             Title used for the plot.

        """

        assert(line.numdims() == 2)
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_plot_nd(self._wnd, line.arr, c_pointer(_cell)))

    def plot3(self, X=None, Y=None, Z=None, line=None, title=None):
        """
        Display a 3D Plot.

        Parameters
        ----------

        line: af.Array.
             - A 3 dimensional array of size [n 3]. Each column denotes X, Y, and Z co-ordinates for plotting 3D lines.

        title: str.
             Title used for the plot.
        """

        assert(line.numdims() == 3)
        _cell = _Cell(self._r, self._c, title, self._cmap)
        safe_call(backend.get().af_draw_plot_nd(self._wnd, line.arr, c_pointer(_cell)))

    def vector_field(self, xpoints, xdirs, ypoints, ydirs, zpoints=None, zdirs=None,
                     points = None, dirs = None, title=None):
        """
        Display a 2D or 3D Vector_Field.

        Parameters
        ----------

        xpoints : af.Array.
                 - A 1 dimensional array containing X co-ordinates.
                 - Not used if points is not None

        xdirs   : af.Array.
                 - A 1 dimensional array specifying direction at current location.
                 - Not used if dirs is not None

        ypoints : af.Array.
                 - A 1 dimensional array containing Y co-ordinates.
                 - Not used if points is not None

        ydirs   : af.Array.
                 - A 1 dimensional array specifying direction at current location.
                 - Not used if dirs is not None

        zpoints : optional: af.Array. default: None.
                 - A 1 dimensional array containing Z co-ordinates.
                 - Not used if points is not None

        zdirs   : optional: af.Array. default: none.
                 - A 1 dimensional array specifying direction at current location.
                 - Not used if dirs is not None

        points  : optional: af.Array. default: None.
             - A 2 dimensional array of size [n 2]. Each column denotes X and Y co-ordinates for plotting 2D lines.
             - A 3 dimensional array of size [n 3]. Each column denotes X, Y, and Z co-ordinates for plotting 3D lines.

        dirs    : optional: af.Array. default: None.
             - A 2 dimensional array of size [n 2]. Each column denotes X and Y directions for plotting 2D lines.
             - A 3 dimensional array of size [n 3]. Each column denotes X, Y, and Z directions for plotting 3D lines.

        title   : str.
             Title used for the plot.

        Note
        ----

        The line parameter takes precedence.
        """
        _cell = _Cell(self._r, self._c, title, self._cmap)
        if line is None:
            if Z is None:
                safe_call(backend.get().af_draw_vector_field_2d(self._wnd,
                                                                xpoints.arr, ypoints.arr,
                                                                xdirs.arr, ydirs.arr,
                                                                c_pointer(_cell)))
            else:
                safe_call(backend.get().af_draw_vector_field_2d(self._wnd,
                                                                xpoints.arr, ypoints.arr, zpoints.arr,
                                                                xdirs.arr, ydirs.arr, zdirs.arr,
                                                                c_pointer(_cell)))
        else:
            safe_call(backend.get().af_draw_plot_nd(self._wnd, points.arr, dirs.arr, c_pointer(_cell)))

    def surface(self, x_vals, y_vals, z_vals, title=None):
        """
        Renders the input array as a 3D surface plot.

        Parameters
        ----------

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
                                                c_pointer(_cell)))

    def hist(self, X, min_val, max_val, title=None):
        """
        Display a histogram Plot.

        Parameters
        ----------

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
                                             c_double_t(max_val), c_double_t(min_val),
                                             c_pointer(_cell)))

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
        safe_call(backend.get().af_grid(self._wnd, c_int_t(rows), c_int_t(cols)))

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
        tmp = c_bool_t(True)
        safe_call(backend.get().af_is_window_closed(c_pointer(tmp), self._wnd))
        return tmp

    def set_visibility(is_visible):
        """
        A flag that shows or hides the window as requested.

        Parameters
        ----------
        is_visible: Flag specifying the visibility of the flag.
        """
        safe_call(backend.get().af_set_visibility(self._wnd, is_visible))

    def set_axes_limits(self, xmin, xmax, ymin, ymax, zmin=None, zmax=None, exact=False):
        """
        Set axis limits.

        Parameters
        ----------

        xmin : af.Array.
              - lower limit of the x axis.

        xmax : af.Array.
              - upper limit of the x axis.

        ymin : af.Array.
              - lower limit of the y axis.

        ymax : af.Array.
              - upper limit of the y axis.

        zmin : optional: af.Array. default: None.
              - lower limit of the z axis.

        zmax : optional: af.Array. default: None.
              - upper limit of the z axis.

        title   : str.
             Title used for the plot.

        Note
        ----

        The line parameter takes precedence.
        """
        _cell = _Cell(self._r, self._c, "", self._cmap)
        if (zmin is None or zmax is None):
            safe_call(backend.get().af_set_axes_limits_2d(self._wnd,
                                                          c_float_t(xmin), c_float_t(xmax),
                                                          c_float_t(ymin), c_float_t(ymax),
                                                          exact, c_pointer(_cell)))
        else:
            safe_call(backend.get().af_set_axes_limits_2d(self._wnd,
                                                          c_float_t(xmin), c_float_t(xmax),
                                                          c_float_t(ymin), c_float_t(ymax),
                                                          c_float_t(zmin), c_float_t(zmax),
                                                          exact, c_pointer(_cell)))

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
