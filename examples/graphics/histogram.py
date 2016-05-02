#!/usr/bin/python

#######################################################
# Copyright (c) 2015, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

import arrayfire as af
import sys
import os

if __name__ == "__main__":

    if (len(sys.argv) == 1):
        raise RuntimeError("Expected to the image as the first argument")

    if not os.path.isfile(sys.argv[1]):
        raise RuntimeError("File %s not found" % sys.argv[1])

    if (len(sys.argv) >  2):
        af.set_device(int(sys.argv[2]))

    af.info()

    hist_win = af.Window(512, 512, "3D Plot example using ArrayFire")
    img_win  = af.Window(480, 640, "Input Image")

    img = af.load_image(sys.argv[1]).as_type(af.Dtype.u8)
    hist = af.histogram(img, 256, 0, 255)

    while (not hist_win.close()) and (not img_win.close()):
        hist_win.hist(hist, 0, 255)
        img_win.image(img)
