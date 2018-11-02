#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file generateTestCase.py
# @brief Generate a sequence of video/x-raw frames
# @todo Generate corresponding golden results
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @url https://github.com/nnsuite/nnstreamer/issues/738


from __future__ import print_function

import sys
import os
import array as arr

##
# @brief Generate a video/x-raw single frame
# @return The frame buffer contents
def genFrame(seq):
    frame = arr.array('B', [0] * 64)
    width = 4
    height = 4
    bpp = 4 # Byte per pixel

    # (ALLCOLOR, 0, 0), 0 to 255
    frame[bpp * ( 0 + width * 0 ) + 0] = seq # B
    frame[bpp * ( 0 + width * 0 ) + 1] = seq # G
    frame[bpp * ( 0 + width * 0 ) + 2] = seq # R

    # (ALLCOLOR, 3, 3), 255 to 0
    frame[bpp * ( 3 + width * 3 ) + 0] = 255 - seq # B
    frame[bpp * ( 3 + width * 3 ) + 1] = 255 - seq # G
    frame[bpp * ( 3 + width * 3 ) + 2] = 255 - seq # R

    # (RED, 1, 1), 0 to 255
    frame[bpp * ( 1 + width * 1 ) + 2] = seq # R

    return frame.tostring()

filename = "video_4x4xBGRx.xraw"
f = open(filename, "w")

for seq in range(0, 256):
    f.write(genFrame(seq))

f.close()

print("File (%s) is written.\n" % filename)
