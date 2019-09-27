#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file generateTestCase.py
# @brief Generate a sequence of video/x-raw frames
# @todo Generate corresponding golden results
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @author Jijoong Moon <jijoong.moon@samsung.com>
# @url https://github.com/nnsuite/nnstreamer/issues/738


from __future__ import print_function

import sys
import os
import array as arr
import struct
import string
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gen24bBMP import convert_to_bytes

def location(c, w, h):
    return c+4*(w+4*h)

##
# @brief Generate a video/x-raw single frame
# @return The frame buffer contents
def genFrame(seq, out):
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

    for h in range(0,4):
        w=0
        for i in range(0,4):
            out[location(i,w,h)] = frame[location(i,w,h)]
        for w in range(1,3):
            for c in range(0,4):
                sum = frame[location(0,w,h)]
                sum += out[location(0,w,h)]
                out[location(0,w,h)] = sum // 2
        w=3
        for i in range(0,4):
            out[location(i,w,h)] = out[location(i,w,h)]

    return frame, out

filename = "video_4x4xBGRx.xraw"
f = open(filename, "wb")

out = arr.array('B',[0]*64)

for seq in range(0, 10):
    outfilename = "rnn.golden"
    frame, out = genFrame(seq,out)

    if(seq == 9):
        with open(outfilename,'wb') as file:
            file.write(out.tostring())

    f.write(frame)

f.close()

print("File (%s) is written.\n" % filename)
