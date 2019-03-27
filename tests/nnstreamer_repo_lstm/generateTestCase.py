#!/usr/bin/env python
##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file generateTestCase.py
# @brief Generate a sequence of video/x-raw frames & golden
# @todo Generate corresponding golden results
# @author Jijoong Moon <jijoong.moon@samsung.com>
# @url https://github.com/nnsuite/nnstreamer/
#

from __future__ import print_function

import sys
import os
import array as arr
import numpy as np

def location(c, w, h):
    return c+4*(w+4*h)

##
# @brief Generate a video/x-raw frame & update results
# @return The frame buffer contents
def genFrame(seq, out0, out1):
    frame = arr.array('f', [0] * 64)
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

    in0 = out0;
    in1 = out1;
    in2 = frame;

    for h in range(0,4):
        for w in range(0,4):
            for c in range(0,4):
                in2_tmp0 = (in2[location(c,w,h)] + in1[location(c,w,h)])/2
                in2_tmp1 = np.tanh((in2[location(c,w,h)]))
                in0[location(c,w,h)] = (in0[location(c,w,h)] * in2_tmp0)
                in0[location(c,w,h)] += (in2_tmp0 * in2_tmp1)
                out0[location(c,w,h)] = in0[location(c,w,h)]
                out1[location(c,w,h)] = (np.tanh(in0[location(c,w,h)]) * in2_tmp0)

    return frame.tostring(), out0, out1

filename = "video_4x4xBGRx.xraw"
f = open(filename, "wb")

out0 = arr.array('f',[0]*64)
out1 = arr.array('f',[0]*64)

for seq in range(0, 10):
    outfilename = "lstm.golden"
    string, out0, out1=genFrame(seq,out0, out1)
    if(seq == 9):
        with open(outfilename,'wb') as file:
            file.write(out1.tostring())

    f.write(string)

f.close()

print("File (%s) is written.\n" % filename)
