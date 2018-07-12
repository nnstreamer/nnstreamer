#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file gen24bBMP.py
# @brief Generate 24bpp .bmp files for test cases
# @author MyungJoo Ham <myungjoo.ham@samsung.com>

from __future__ import print_function

from struct import pack
import sys


##
# @brief Save bitmap "data" to "filename"
# @param[in] filename The filename to be saves as a .png file.
# @param[in] data string of RGB (packed in 'BBB') or BGRx (packed in 'BBBB')
# @param[in] colorspace "RGB" or "BGRx"
# @param[in] width Width of the picture
# @param[in] height Height of the picture
def saveBMP(filename, data, colorspace, width, height):
    size = len(data)
    graphics = ""

    if colorspace == 'RGB':
        assert(size == (width * height * 3))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = 3 * (w + width * h)
                graphics += data[pos+2]
                graphics += data[pos+1]
                graphics += data[pos]
            for x in range(0, (width * 3) % 4):
                graphics += pack('<B', 0)
    elif colorspace == 'BGRx':
        assert(size == (width * height * 4))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = 4 * (w + width * h)
                graphics += data[pos]
                graphics += data[pos+1]
                graphics += data[pos+2]
            for x in range(0, (width * 3) % 4):
                graphics += pack('<B', 0)
    else:
        print("Unrecognized colorspace %", colorspace)
        sys.exit(1)

    # BMP file header
    header = ""
    header += pack('<HLHHL', 19778, (26 + width * height * 3), 0, 0, 26)
    header += pack('<LHHHH', 12, width, height, 1, 24)

    file = open(filename, 'wb')
    file.write(header)
    file.write(graphics)
    file.close()
