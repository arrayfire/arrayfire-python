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

def draw_corners(img, x, y, draw_len):
    # Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
    # Set only the first channel to 1 (green lines)
    xmin = max(0, x - draw_len)
    xmax = min(img.dims()[1], x + draw_len)

    img[y, xmin : xmax, 0] = 0.0
    img[y, xmin : xmax, 1] = 1.0
    img[y, xmin : xmax, 2] = 0.0

    # Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
    # Set only the first channel to 1 (green lines)
    ymin = int(max(0, y - draw_len))
    ymax = int(min(img.dims()[0], y + draw_len))

    img[ymin : ymax, x, 0] = 0.0
    img[ymin : ymax, x, 1] = 1.0
    img[ymin : ymax, x, 2] = 0.0
    return img

def fast_demo(console):

    root_path = os.path.dirname(os.path.abspath(__file__))
    file_path = root_path
    if console:
        file_path += "/../../assets/examples/images/square.png"
    else:
        file_path += "/../../assets/examples/images/man.jpg"
    img_color = af.load_image(file_path, True);

    img = af.color_space(img_color, af.CSPACE.GRAY, af.CSPACE.RGB)
    img_color /= 255.0

    features = af.fast(img)

    xs = features.get_xpos().to_list()
    ys = features.get_ypos().to_list()

    draw_len = 3;
    num_features = features.num_features().value
    for f in range(num_features):
        print(f)
        x = xs[f]
        y = ys[f]

        img_color = draw_corners(img_color, x, y, draw_len)


    print("Features found: {}".format(num_features))
    if not console:
        # Previews color image with green crosshairs
        wnd = af.Window(512, 512, "FAST Feature Detector")

        while not wnd.close():
            wnd.image(img_color)
    else:
        print(xs);
        print(ys);


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        af.set_device(int(sys.argv[1]))
    console = (sys.argv[2] == '-') if len(sys.argv) > 2 else False

    af.info()
    print("** ArrayFire FAST Feature Detector Demo **\n")
    fast_demo(console)

