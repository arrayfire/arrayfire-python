#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

"""
Image processing functions.
"""

from .library import *
from .array import *
from .data import constant
from .signal import medfilt
import os

def gradient(image):
    """
    Find the horizontal and vertical gradients.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    Returns
    ---------
    (dx, dy) : Tuple of af.Array.
             - `dx` containing the horizontal gradients of `image`.
             - `dy` containing the vertical gradients of `image`.

    """
    dx = Array()
    dy = Array()
    safe_call(backend.get().af_gradient(c_pointer(dx.arr), c_pointer(dy.arr), image.arr))
    return dx, dy

def load_image(file_name, is_color=False):
    """
    Load an image on the disk as an array.

    Parameters
    ----------
    file_name: str
          - Full path of the file name on disk.

    is_color : optional: bool. default: False.
          - Specifies if the image is loaded as 1 channel (if False) or 3 channel image (if True).

    Returns
    -------
    image - af.Array
            A 2 dimensional (1 channel) or 3 dimensional (3 channel) array containing the image.

    """
    assert(os.path.isfile(file_name))
    image = Array()
    safe_call(backend.get().af_load_image(c_pointer(image.arr),
                                          c_char_ptr_t(file_name.encode('ascii')), is_color))
    return image

def save_image(image, file_name):
    """
    Save an array as an image on the disk.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image.

    file_name: str
          - Full path of the file name on the disk.
    """
    assert(isinstance(file_name, str))
    safe_call(backend.get().af_save_image(c_char_ptr_t(file_name.encode('ascii')), image.arr))
    return image


def load_image_native(file_name):
    """
    Load an image on the disk as an array in native format.

    Parameters
    ----------
    file_name: str
          - Full path of the file name on disk.

    Returns
    -------
    image - af.Array
            A 2 dimensional (1 channel) or 3 dimensional (3 or 4 channel) array containing the image.

    """
    assert(os.path.isfile(file_name))
    image = Array()
    safe_call(backend.get().af_load_image_native(c_pointer(image.arr),
                                                 c_char_ptr_t(file_name.encode('ascii'))))
    return image

def save_image_native(image, file_name):
    """
    Save an array as an image on the disk in native format.

    Parameters
    ----------
    image : af.Array
          - A 2 or 3 dimensional arrayfire array representing an image.

    file_name: str
          - Full path of the file name on the disk.
    """
    assert(isinstance(file_name, str))
    safe_call(backend.get().af_save_image_native(c_char_ptr_t(file_name.encode('ascii')), image.arr))
    return image

def resize(image, scale=None, odim0=None, odim1=None, method=INTERP.NEAREST):
    """
    Resize an image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    scale : optional: scalar. default: None.
          - Scale factor for the image resizing.

    odim0 : optional: int. default: None.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: None.
          - Size of the second dimension of the output.

    method : optional: af.INTERP. default: af.INTERP.NEAREST.
          - Interpolation method used for resizing.

    Returns
    ---------
    out  : af.Array
          - Output image after resizing.

    Note
    -----

    - If `scale` is None, `odim0` and `odim1` need to be specified.
    - If `scale` is not None, `odim0` and `odim1` are ignored.

    """
    if (scale is None):
        assert(odim0 is not None)
        assert(odim1 is not None)
    else:
        idims = image.dims()
        odim0 = int(scale * idims[0])
        odim1 = int(scale * idims[1])

    output = Array()
    safe_call(backend.get().af_resize(c_pointer(output.arr),
                                      image.arr, c_dim_t(odim0),
                                      c_dim_t(odim1), method.value))

    return output

def transform(image, trans_mat, odim0 = 0, odim1 = 0, method=INTERP.NEAREST, is_inverse=True):
    """
    Transform an image using a transformation matrix.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    trans_mat : af.Array
          - A 2 D floating point arrayfire array of size [3, 2].

    odim0 : optional: int. default: 0.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: 0.
          - Size of the second dimension of the output.

    method : optional: af.INTERP. default: af.INTERP.NEAREST.
          - Interpolation method used for transformation.

    is_inverse : optional: bool. default: True.
          - Specifies if the inverse transform is applied.

    Returns
    ---------
    out  : af.Array
          - Output image after transformation.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().af_transform(c_pointer(output.arr),
                                         image.arr, trans_mat.arr,
                                         c_dim_t(odim0), c_dim_t(odim1),
                                         method.value, is_inverse))
    return output


def rotate(image, theta, is_crop = True, method = INTERP.NEAREST):
    """
    Rotate an image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    theta : scalar
          - The angle to rotate in radians.

    is_crop : optional: bool. default: True.
          - Specifies if the output should be cropped to the input size.

    method : optional: af.INTERP. default: af.INTERP.NEAREST.
          - Interpolation method used for rotating.

    Returns
    ---------
    out  : af.Array
          - Output image after rotating.
    """
    output = Array()
    safe_call(backend.get().af_rotate(c_pointer(output.arr), image.arr,
                                      c_float_t(theta), is_crop, method.value))
    return output

def translate(image, trans0, trans1, odim0 = 0, odim1 = 0, method = INTERP.NEAREST):
    """
    Translate an image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    trans0: int.
          - Translation along first dimension in pixels.

    trans1: int.
          - Translation along second dimension in pixels.

    odim0 : optional: int. default: 0.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: 0.
          - Size of the second dimension of the output.

    method : optional: af.INTERP. default: af.INTERP.NEAREST.
          - Interpolation method used for translation.

    Returns
    ---------
    out  : af.Array
          - Output image after translation.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().af_translate(c_pointer(output.arr),
                                         image.arr, trans0, trans1,
                                         c_dim_t(odim0), c_dim_t(odim1), method.value))
    return output

def scale(image, scale0, scale1, odim0 = 0, odim1 = 0, method = INTERP.NEAREST):
    """
    Scale an image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    scale0 : scalar.
          - Scale factor for the first dimension.

    scale1 : scalar.
          - Scale factor for the second dimension.

    odim0 : optional: int. default: None.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: None.
          - Size of the second dimension of the output.

    method : optional: af.INTERP. default: af.INTERP.NEAREST.
          - Interpolation method used for resizing.

    Returns
    ---------
    out  : af.Array
          - Output image after scaling.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().af_scale(c_pointer(output.arr),
                                     image.arr, c_float_t(scale0), c_float_t(scale1),
                                     c_dim_t(odim0), c_dim_t(odim1), method.value))
    return output

def skew(image, skew0, skew1, odim0 = 0, odim1 = 0, method = INTERP.NEAREST, is_inverse=True):
    """
    Skew an image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    skew0 : scalar.
          - Skew factor for the first dimension.

    skew1 : scalar.
          - Skew factor for the second dimension.

    odim0 : optional: int. default: None.
          - Size of the first dimension of the output.

    odim1 : optional: int. default: None.
          - Size of the second dimension of the output.

    method : optional: af.INTERP. default: af.INTERP.NEAREST.
          - Interpolation method used for resizing.

    is_inverse : optional: bool. default: True.
          - Specifies if the inverse skew  is applied.

    Returns
    ---------
    out  : af.Array
          - Output image after skewing.

    Note
    -----

    - If `odim0` and `odim` are 0, the output dimensions are automatically calculated by the function.

    """
    output = Array()
    safe_call(backend.get().af_skew(c_pointer(output.arr),
                                    image.arr, c_float_t(skew0), c_float_t(skew1),
                                    c_dim_t(odim0), c_dim_t(odim1),
                                    method.value, is_inverse))

    return output

def histogram(image, nbins, min_val = None, max_val = None):
    """
    Find the histogram of an image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    nbins : int.
          - Number of bins in the histogram.

    min_val : optional: scalar. default: None.
          - The lower bound for the bin values.
          - If None, `af.min(image)` is used.

    max_val : optional: scalar. default: None.
          - The upper bound for the bin values.
          - If None, `af.max(image)` is used.

    Returns
    ---------
    hist : af.Array
          - Containing the histogram of the image.

    """
    from .algorithm import min as af_min
    from .algorithm import max as af_max

    if min_val is None:
        min_val = af_min(image)

    if max_val is None:
        max_val = af_max(image)

    output = Array()
    safe_call(backend.get().af_histogram(c_pointer(output.arr),
                                         image.arr, c_uint_t(nbins),
                                         c_double_t(min_val), c_double_t(max_val)))
    return output

def hist_equal(image, hist):
    """
    Equalize an image based on a histogram.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    hist : af.Array
          - Containing the histogram of an image.

    Returns
    ---------

    output : af.Array
           - The equalized image.

    """
    output = Array()
    safe_call(backend.get().af_hist_equal(c_pointer(output.arr), image.arr, hist.arr))
    return output

def dilate(image, mask = None):
    """
    Run image dilate on the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    mask  : optional: af.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3] array of all ones is used.

    Returns
    ---------

    output : af.Array
           - The dilated image.

    """
    if mask is None:
        mask = constant(1, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().af_dilate(c_pointer(output.arr), image.arr, mask.arr))

    return output

def dilate3(volume, mask = None):
    """
    Run volume dilate on a volume.

    Parameters
    ----------
    volume : af.Array
          - A 3 D arrayfire array representing a volume, or
          - A multi dimensional array representing batch of volumes.

    mask  : optional: af.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3, 3] array of all ones is used.

    Returns
    ---------

    output : af.Array
           - The dilated volume.

    """
    if mask is None:
        mask = constant(1, 3, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().af_dilate3(c_pointer(output.arr), volume.arr, mask.arr))

    return output

def erode(image, mask = None):
    """
    Run image erode on the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    mask  : optional: af.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3] array of all ones is used.

    Returns
    ---------

    output : af.Array
           - The eroded image.

    """
    if mask is None:
        mask = constant(1, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().af_erode(c_pointer(output.arr), image.arr, mask.arr))

    return output

def erode3(volume, mask = None):
    """
    Run volume erode on the volume.

    Parameters
    ----------
    volume : af.Array
          - A 3 D arrayfire array representing an volume, or
          - A multi dimensional array representing batch of volumes.

    mask  : optional: af.Array. default: None.
          - Specifies the neighborhood of a pixel.
          - When None, a [3, 3, 3] array of all ones is used.

    Returns
    ---------

    output : af.Array
           - The eroded volume.

    """

    if mask is None:
        mask = constant(1, 3, 3, 3, dtype=Dtype.f32)

    output = Array()
    safe_call(backend.get().af_erode3(c_pointer(output.arr), volume.arr, mask.arr))

    return output

def bilateral(image, s_sigma, c_sigma, is_color = False):
    """
    Apply bilateral filter to the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    s_sigma : scalar.
          - Sigma value for the co-ordinate space.

    c_sigma : scalar.
          - Sigma value for the color space.

    is_color : optional: bool. default: False.
          - Specifies if the third dimension is 3rd channel (if True) or a batch (if False).

    Returns
    ---------

    output : af.Array
           - The image after the application of the bilateral filter.

    """
    output = Array()
    safe_call(backend.get().af_bilateral(c_pointer(output.arr),
                                         image.arr, c_float_t(s_sigma),
                                         c_float_t(c_sigma), is_color))
    return output

def mean_shift(image, s_sigma, c_sigma, n_iter, is_color = False):
    """
    Apply mean shift to the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    s_sigma : scalar.
          - Sigma value for the co-ordinate space.

    c_sigma : scalar.
          - Sigma value for the color space.

    n_iter  : int.
          - Number of mean shift iterations.

    is_color : optional: bool. default: False.
          - Specifies if the third dimension is 3rd channel (if True) or a batch (if False).

    Returns
    ---------

    output : af.Array
           - The image after the application of the meanshift.

    """
    output = Array()
    safe_call(backend.get().af_mean_shift(c_pointer(output.arr),
                                          image.arr, c_float_t(s_sigma), c_float_t(c_sigma),
                                          c_uint_t(n_iter), is_color))
    return output

def minfilt(image, w_len = 3, w_wid = 3, edge_pad = PAD.ZERO):
    """
    Apply min filter for the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    w0 : optional: int. default: 3.
          - The length of the filter along the first dimension.

    w1 : optional: int. default: 3.
          - The length of the filter along the second dimension.

    edge_pad : optional: af.PAD. default: af.PAD.ZERO
          - Flag specifying how the min at the edge should be treated.

    Returns
    ---------

    output : af.Array
           - The image after min filter is applied.

    """
    output = Array()
    safe_call(backend.get().af_minfilt(c_pointer(output.arr),
                                       image.arr, c_dim_t(w_len),
                                       c_dim_t(w_wid), edge_pad.value))
    return output

def maxfilt(image, w_len = 3, w_wid = 3, edge_pad = PAD.ZERO):
    """
    Apply max filter for the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    w0 : optional: int. default: 3.
          - The length of the filter along the first dimension.

    w1 : optional: int. default: 3.
          - The length of the filter along the second dimension.

    edge_pad : optional: af.PAD. default: af.PAD.ZERO
          - Flag specifying how the max at the edge should be treated.

    Returns
    ---------

    output : af.Array
           - The image after max filter is applied.

    """
    output = Array()
    safe_call(backend.get().af_maxfilt(c_pointer(output.arr),
                                       image.arr, c_dim_t(w_len),
                                       c_dim_t(w_wid), edge_pad.value))
    return output

def regions(image, conn = CONNECTIVITY.FOUR, out_type = Dtype.f32):
    """
    Find the connected components in the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image.

    conn : optional: af.CONNECTIVITY. default: af.CONNECTIVITY.FOUR.
          - Specifies the connectivity of the pixels.

    out_type : optional: af.Dtype. default: af.Dtype.f32.
          - Specifies the type for the output.

    Returns
    ---------

    output : af.Array
           - An array where each pixel is labeled with its component number.

    """
    output = Array()
    safe_call(backend.get().af_regions(c_pointer(output.arr), image.arr,
                                       conn.value, out_type.value))
    return output

def sobel_derivatives(image, w_len=3):
    """
    Find the sobel derivatives of the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    w_len : optional: int. default: 3.
          - The size of the sobel operator.

    Returns
    ---------

    (dx, dy) : tuple of af.Arrays.
           - `dx` is the sobel derivative along the horizontal direction.
           - `dy` is the sobel derivative along the vertical direction.

    """
    dx = Array()
    dy = Array()
    safe_call(backend.get().af_sobel_operator(c_pointer(dx.arr), c_pointer(dy.arr),
                                              image.arr, c_uint_t(w_len)))
    return dx,dy

def gaussian_kernel(rows, cols, sigma_r = None, sigma_c = None):
    """
    Create a gaussian kernel with the given parameters.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    rows : int
         - The number of rows in the gaussian kernel.

    cols : int
         - The number of columns in the gaussian kernel.

    sigma_r : optional: number. default: None.
         - The sigma value along rows
         - If None, calculated as (0.25 * rows + 0.75)

    sigma_c : optional: number. default: None.
         - The sigma value along columns
         - If None, calculated as (0.25 * cols + 0.75)

    Returns
    -------
    out   : af.Array
          - A gaussian kernel of size (rows, cols)
    """
    out = Array()

    if (sigma_r is None):
        sigma_r = 0.25 * rows + 0.75

    if (sigma_c is None):
        sigma_c = 0.25 * cols + 0.75

    safe_call(backend.get().af_gaussian_kernel(c_pointer(out.arr),
                                               c_int_t(rows), c_int_t(cols),
                                               c_double_t(sigma_r), c_double_t(sigma_c)))
    return out

def sobel_filter(image, w_len = 3, is_fast = False):
    """
    Apply sobel filter to the image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    w_len : optional: int. default: 3.
          - The size of the sobel operator.

    is_fast : optional: bool. default: False.
          - Specifies if the magnitude is generated using SAD (if True) or SSD (if False).

    Returns
    ---------

    output : af.Array
           - Image containing the magnitude of the sobel derivatives.

    """
    from .arith import abs as af_abs
    from .arith import hypot as af_hypot

    dx,dy = sobel_derivatives(image, w_len)
    if (is_fast):
        return af_abs(dx) + af_abs(dy)
    else:
        return af_hypot(dx, dy)

def rgb2gray(image, r_factor = 0.2126, g_factor = 0.7152, b_factor = 0.0722):
    """
    Convert RGB image to Grayscale.

    Parameters
    ----------
    image : af.Array
          - A 3 D arrayfire array representing an 3 channel image, or
          - A multi dimensional array representing batch of images.

    r_factor : optional: scalar. default: 0.2126.
          - Weight for the red channel.

    g_factor : optional: scalar. default: 0.7152.
          - Weight for the green channel.

    b_factor : optional: scalar. default: 0.0722.
          - Weight for the blue channel.

    Returns
    --------

    output : af.Array
          - A grayscale image.

    """
    output=Array()
    safe_call(backend.get().af_rgb2gray(c_pointer(output.arr),
                                        image.arr, c_float_t(r_factor), c_float_t(g_factor), c_float_t(b_factor)))
    return output

def gray2rgb(image, r_factor = 1.0, g_factor = 1.0, b_factor = 1.0):
    """
    Convert Grayscale image to an RGB image.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    r_factor : optional: scalar. default: 1.0.
          - Scale factor for the red channel.

    g_factor : optional: scalar. default: 1.0.
          - Scale factor for the green channel.

    b_factor : optional: scalar. default: 1.0
          - Scale factor for the blue channel.

    Returns
    --------

    output : af.Array
          - An RGB image.
          - The channels are not coalesced, i.e. they appear along the third dimension.

    """
    output=Array()
    safe_call(backend.get().af_gray2rgb(c_pointer(output.arr),
                                        image.arr, c_float_t(r_factor), c_float_t(g_factor), c_float_t(b_factor)))
    return output

def hsv2rgb(image):
    """
    Convert HSV image to RGB.

    Parameters
    ----------
    image : af.Array
          - A 3 D arrayfire array representing an 3 channel image, or
          - A multi dimensional array representing batch of images.

    Returns
    --------

    output : af.Array
          - A HSV image.

    """
    output = Array()
    safe_call(backend.get().af_hsv2rgb(c_pointer(output.arr), image.arr))
    return output

def rgb2hsv(image):
    """
    Convert RGB image to HSV.

    Parameters
    ----------
    image : af.Array
          - A 3 D arrayfire array representing an 3 channel image, or
          - A multi dimensional array representing batch of images.

    Returns
    --------

    output : af.Array
          - A RGB image.

    """
    output = Array()
    safe_call(backend.get().af_rgb2hsv(c_pointer(output.arr), image.arr))
    return output

def color_space(image, to_type, from_type):
    """
    Convert an image from one color space to another.

    Parameters
    ----------
    image : af.Array
          - A multi dimensional array representing batch of images in `from_type` color space.

    to_type : af.CSPACE
          - An enum for the destination color space.

    from_type : af.CSPACE
          - An enum for the source color space.

    Returns
    --------

    output : af.Array
          - An image in the `to_type` color space.

    """
    output = Array()
    safe_call(backend.get().af_color_space(c_pointer(output.arr), image.arr,
                                           to_type.value, from_type.value))
    return output

def unwrap(image, wx, wy, sx, sy, px=0, py=0, is_column=True):
    """
    Unrwap an image into an array.

    Parameters
    ----------

    image  : af.Array
           A multi dimensional array specifying an image or batch of images.

    wx     : Integer.
           Block window size along the first dimension.

    wy     : Integer.
           Block window size along the second dimension.

    sx     : Integer.
           Stride along the first dimension.

    sy     : Integer.
           Stride along the second dimension.

    px     : Integer. Optional. Default: 0
           Padding along the first dimension.

    py     : Integer. Optional. Default: 0
           Padding along the second dimension.

    is_column : Boolean. Optional. Default: True.
           Specifies if the patch should be laid along row or columns.

    Returns
    -------

    out   : af.Array
          A multi dimensional array contianing the image patches along specified dimension.

    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(6, 6)
    >>> af.display(a)

    [6 6 1 1]
        0.4107     0.3775     0.0901     0.8060     0.0012     0.9250
        0.8224     0.3027     0.5933     0.5938     0.8703     0.3063
        0.9518     0.6456     0.1098     0.8395     0.5259     0.9313
        0.1794     0.5591     0.1046     0.1933     0.1443     0.8684
        0.4198     0.6600     0.8827     0.7270     0.3253     0.6592
        0.0081     0.0764     0.1647     0.0322     0.5081     0.4387

    >>> b = af.unwrap(a, 2, 2, 2, 2)
    >>> af.display(b)

    [4 9 1 1]
        0.4107     0.9518     0.4198     0.0901     0.1098     0.8827     0.0012     0.5259     0.3253
        0.8224     0.1794     0.0081     0.5933     0.1046     0.1647     0.8703     0.1443     0.5081
        0.3775     0.6456     0.6600     0.8060     0.8395     0.7270     0.9250     0.9313     0.6592
        0.3027     0.5591     0.0764     0.5938     0.1933     0.0322     0.3063     0.8684     0.4387
    """

    out = Array()
    safe_call(backend.get().af_unwrap(c_pointer(out.arr), image.arr,
                                      c_dim_t(wx), c_dim_t(wy),
                                      c_dim_t(sx), c_dim_t(sy),
                                      c_dim_t(px), c_dim_t(py),
                                      is_column))
    return out

def wrap(a, ox, oy, wx, wy, sx, sy, px=0, py=0, is_column=True):
    """
    Wrap an array into an image.

    Parameters
    ----------

    a      : af.Array
           A multi dimensional array containing patches of images.

    wx     : Integer.
           Block window size along the first dimension.

    wy     : Integer.
           Block window size along the second dimension.

    sx     : Integer.
           Stride along the first dimension.

    sy     : Integer.
           Stride along the second dimension.

    px     : Integer. Optional. Default: 0
           Padding along the first dimension.

    py     : Integer. Optional. Default: 0
           Padding along the second dimension.

    is_column : Boolean. Optional. Default: True.
           Specifies if the patch should be laid along row or columns.

    Returns
    -------

    out   : af.Array
          A multi dimensional array contianing the images.


    Examples
    --------
    >>> import arrayfire as af
    >>> a = af.randu(6, 6)
    >>> af.display(a)

    [6 6 1 1]
        0.4107     0.3775     0.0901     0.8060     0.0012     0.9250
        0.8224     0.3027     0.5933     0.5938     0.8703     0.3063
        0.9518     0.6456     0.1098     0.8395     0.5259     0.9313
        0.1794     0.5591     0.1046     0.1933     0.1443     0.8684
        0.4198     0.6600     0.8827     0.7270     0.3253     0.6592
        0.0081     0.0764     0.1647     0.0322     0.5081     0.4387

    >>> b = af.unwrap(a, 2, 2, 2, 2)
    >>> af.display(b)

    [4 9 1 1]
        0.4107     0.9518     0.4198     0.0901     0.1098     0.8827     0.0012     0.5259     0.3253
        0.8224     0.1794     0.0081     0.5933     0.1046     0.1647     0.8703     0.1443     0.5081
        0.3775     0.6456     0.6600     0.8060     0.8395     0.7270     0.9250     0.9313     0.6592
        0.3027     0.5591     0.0764     0.5938     0.1933     0.0322     0.3063     0.8684     0.4387

    >>> af.display(c)

    [6 6 1 1]
        0.4107     0.3775     0.0901     0.8060     0.0012     0.9250
        0.8224     0.3027     0.5933     0.5938     0.8703     0.3063
        0.9518     0.6456     0.1098     0.8395     0.5259     0.9313
        0.1794     0.5591     0.1046     0.1933     0.1443     0.8684
        0.4198     0.6600     0.8827     0.7270     0.3253     0.6592
        0.0081     0.0764     0.1647     0.0322     0.5081     0.4387


    """

    out = Array()
    safe_call(backend.get().af_wrap(c_pointer(out.arr), a.arr,
                                    c_dim_t(ox), c_dim_t(oy),
                                    c_dim_t(wx), c_dim_t(wy),
                                    c_dim_t(sx), c_dim_t(sy),
                                    c_dim_t(px), c_dim_t(py),
                                    is_column))
    return out

def sat(image):
    """
    Summed Area Tables

    Parameters
    ----------
    image : af.Array
          A multi dimensional array specifying image or batch of images

    Returns
    -------
    out  : af.Array
         A multi dimensional array containing the summed area table of input image
    """

    out = Array()
    safe_call(backend.get().af_sat(c_pointer(out.arr), image.arr))
    return out

def ycbcr2rgb(image, standard=YCC_STD.BT_601):
    """
    YCbCr to RGB colorspace conversion.

    Parameters
    ----------

    image   : af.Array
              A multi dimensional array containing an image or batch of images in YCbCr format.

    standard: YCC_STD. optional. default: YCC_STD.BT_601
            - Specifies the YCbCr format.
            - Can be one of YCC_STD.BT_601, YCC_STD.BT_709, and YCC_STD.BT_2020.

    Returns
    --------

    out     : af.Array
            A multi dimensional array containing an image or batch of images in RGB format

    """

    out = Array()
    safe_call(backend.get().af_ycbcr2rgb(c_pointer(out.arr), image.arr, standard.value))
    return out

def rgb2ycbcr(image, standard=YCC_STD.BT_601):
    """
    RGB to YCbCr colorspace conversion.

    Parameters
    ----------

    image   : af.Array
              A multi dimensional array containing an image or batch of images in RGB format.

    standard: YCC_STD. optional. default: YCC_STD.BT_601
            - Specifies the YCbCr format.
            - Can be one of YCC_STD.BT_601, YCC_STD.BT_709, and YCC_STD.BT_2020.

    Returns
    --------

    out     : af.Array
            A multi dimensional array containing an image or batch of images in YCbCr format

    """

    out = Array()
    safe_call(backend.get().af_rgb2ycbcr(c_pointer(out.arr), image.arr, standard.value))
    return out

def moments(image, moment = MOMENT.FIRST_ORDER):
    """
    Calculate image moments.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image, or
          - A multi dimensional array representing batch of images.

    moment : optional: af.MOMENT. default: af.MOMENT.FIRST_ORDER.
          Moment(s) to calculate. Can be one of:
          - af.MOMENT.M00
          - af.MOMENT.M01
          - af.MOMENT.M10
          - af.MOMENT.M11
          - af.MOMENT.FIRST_ORDER

    Returns
    ---------
    out  : af.Array
          - array containing requested moment(s) of each image
    """
    output = Array()
    safe_call(backend.get().af_moments(c_pointer(output.arr), image.arr, moment.value))
    return output

def canny(image,
          low_threshold, high_threshold = None,
          threshold_type = CANNY_THRESHOLD.MANUAL,
          sobel_window = 3, is_fast = False):
    """
    Canny edge detector.

    Parameters
    ----------
    image : af.Array
          - A 2 D arrayfire array representing an image

    threshold_type : optional: af.CANNY_THRESHOLD. default: af.CANNY_THRESHOLD.MANUAL.
          Can be one of:
          - af.CANNY_THRESHOLD.MANUAL
          - af.CANNY_THRESHOLD.AUTO_OTSU

    low_threshold :  required: float.
          Specifies the % of maximum in gradient image if threshold_type is MANUAL.
          Specifies the % of auto dervied high value if threshold_type is AUTO_OTSU.

    high_threshold : optional: float. default: None
          Specifies the % of maximum in gradient image if threshold_type is MANUAL.
          Ignored if threshold_type is AUTO_OTSU

    sobel_window : optional: int. default: 3
          Specifies the size of sobel kernel when computing the gradient image.

    Returns
    --------

    out : af.Array
        - A binary image containing the edges

    """
    output = Array()
    if threshold_type.value == CANNY_THRESHOLD.MANUAL.value:
        assert(high_threshold is not None)

    high_threshold = high_threshold if high_threshold else 0
    safe_call(backend.get().af_canny(c_pointer(output.arr), image.arr,
                                     c_int_t(threshold_type.value),
                                     c_float_t(low_threshold), c_float_t(high_threshold),
                                     c_uint_t(sobel_window), c_bool_t(is_fast)))
    return output

def anisotropic_diffusion(image, time_step, conductance, iterations, flux_function_type = FLUX.QUADRATIC, diffusion_kind = DIFFUSION.GRAD):
    """
    Anisotropic smoothing filter.

    Parameters
    ----------
    image: af.Array
        The input image.

    time_step: scalar.
        The time step used in solving the diffusion equation.

    conductance:
        Controls conductance sensitivity in diffusion equation.

    iterations:
        Number of times the diffusion step is performed.

    flux_function_type:
        Type of flux function to be used. Available flux functions:
          - Quadratic (af.FLUX.QUADRATIC)
          - Exponential (af.FLUX.EXPONENTIAL)

    diffusion_kind:
        Type of diffusion equatoin to be used. Available diffusion equations:
          - Gradient diffusion equation (af.DIFFUSION.GRAD)
          - Modified curvature diffusion equation (af.DIFFUSION.MCDE)

    Returns
    -------
    out: af.Array
        Anisotropically-smoothed output image.

    """
    out = Array()
    safe_call(backend.get().
              af_anisotropic_diffusion(c_pointer(out.arr), image.arr,
                                       c_float_t(time_step), c_float_t(conductance), c_uint_t(iterations),
                                       flux_function_type.value, diffusion_kind.value))
    return out

def is_image_io_available():
    """
    Function to check if the arrayfire library was built with Image IO support.
    """
    res = c_bool_t(False)
    safe_call(backend.get().af_is_image_io_available(c_pointer(res)))
    return res.value
