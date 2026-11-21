#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file generateTest.py
# @brief Generate the snpe-depth and snpe-deeplab input tensors and the
#        snpe-depth golden frame for the image_segment SSAT cases.
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date 22 Nov 2026
# @bug No known bugs

import struct

# 257 x 255 = 65535 pixels, not a multiple of four
WIDTH = 257
HEIGHT = 255

# Values of a snpe-deeplab label map: background, labels within the default
# 20 labels (2.5 must take the color of 2), and labels past the color map
DEEPLAB_VALUES = [0.0, 1.0, 2.0, 2.5, 7.0, 20.0, 21.0, 255.0]


##
# @brief Write the snpe-depth input and its golden RGBA frame.
#
# Every pixel is an integer gray value in [0, 255] except the last one, the
# only 256 and so the maximum. Dividing by a power of two is exact in float32,
# so the decoder draws floor(v * 255 / 256) on every channel of a pixel.
def write_snpe_depth():
    values = [float((x * 7 + y * 13) % 256)
              for y in range(HEIGHT) for x in range(WIDTH)]
    values[-1] = 256.0

    golden = bytearray()
    for value in values:
        gray = int(value) * 255 // 256
        golden += bytes([gray, gray, gray, 0xFF])

    with open("snpe_depth.raw", "wb") as f:
        f.write(struct.pack("<%df" % len(values), *values))
    with open("snpe_depth.golden", "wb") as f:
        f.write(golden)


##
# @brief Write the snpe-deeplab input, a label map with a label on the last pixel.
def write_snpe_deeplab():
    values = [DEEPLAB_VALUES[(x + 3 * y) % len(DEEPLAB_VALUES)]
              for y in range(HEIGHT) for x in range(WIDTH)]
    assert values[-1] == 2.0

    with open("snpe_deeplab.raw", "wb") as f:
        f.write(struct.pack("<%df" % len(values), *values))


if __name__ == "__main__":
    write_snpe_depth()
    write_snpe_deeplab()
