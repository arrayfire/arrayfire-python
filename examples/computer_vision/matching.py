#!/usr/bin/env python

#######################################################
# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from time import time
import arrayfire as af
import os
import sys

def normalize(a):
    max_ = float(af.max(a))
    min_ = float(af.min(a))
    return  (a - min_) /  (max_ - min_)

def draw_rectangle(img, x, y, wx, wy):
    print("\nMatching patch origin = ({}, {})\n".format(x, y))

    # top edge
    img[y, x : x + wx, 0] = 0.0
    img[y, x : x + wx, 1] = 0.0
    img[y, x : x + wx, 2] = 1.0

    # bottom edge
    img[y + wy, x : x + wx, 0] = 0.0
    img[y + wy, x : x + wx, 1] = 0.0
    img[y + wy, x : x + wx, 2] = 1.0

    # left edge
    img[y : y + wy, x, 0] = 0.0
    img[y : y + wy, x, 1] = 0.0
    img[y : y + wy, x, 2] = 1.0

    # left edge
    img[y : y + wy, x + wx, 0] = 0.0
    img[y : y + wy, x + wx, 1] = 0.0
    img[y : y + wy, x + wx, 2] = 1.0

    return img

def templateMatchingDemo(console):

    root_path = os.path.dirname(os.path.abspath(__file__))
    file_path = root_path
    if console:
        file_path += "/../../assets/examples/images/square.png"
    else:
        file_path += "/../../assets/examples/images/man.jpg"
    img_color = af.load_image(file_path, True);

    # Convert the image from RGB to gray-scale
    img = af.color_space(img_color, af.CSPACE.GRAY, af.CSPACE.RGB)
    iDims = img.dims()
    print("Input image dimensions: ", iDims)

    # Extract a patch from the input image
    patch_size = 100
    tmp_img = img[100 : 100+patch_size, 100 : 100+patch_size]

    result = af.match_template(img, tmp_img) # Default disparity metric is
                                             # Sum of Absolute differences (SAD)
                                             # Currently supported metrics are
                                             # AF_SAD, AF_ZSAD, AF_LSAD, AF_SSD,
                                             # AF_ZSSD, AF_LSSD

    disp_img = img / 255.0
    disp_tmp = tmp_img / 255.0
    disp_res = normalize(result)

    minval, minloc = af.imin(disp_res)
    print("Location(linear index) of minimum disparity value = {}".format(minloc))

    if not console:
        marked_res = af.tile(disp_img, 1, 1, 3)
        marked_res = draw_rectangle(marked_res, minloc%iDims[0], minloc/iDims[0],\
                                    patch_size, patch_size)

        print("Note: Based on the disparity metric option provided to matchTemplate function")
        print("either minimum or maximum disparity location is the starting corner")
        print("of our best matching patch to template image in the search image")

        wnd = af.Window(512, 512, "Template Matching Demo")

        while not wnd.close():
            wnd.set_colormap(af.COLORMAP.DEFAULT)
            wnd.grid(2, 2)
            wnd[0, 0].image(disp_img, "Search Image" )
            wnd[0, 1].image(disp_tmp, "Template Patch" )
            wnd[1, 0].image(marked_res, "Best Match" )
            wnd.set_colormap(af.COLORMAP.HEAT)
            wnd[1, 1].image(disp_res, "Disparity Values")
            wnd.show()


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    console = (sys.argv[2] == '-') if len(sys.argv) > 2 else False

    af.info()
    print("** ArrayFire template matching Demo **\n")
    templateMatchingDemo(console)

