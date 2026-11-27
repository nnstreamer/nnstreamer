#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file checkResult.py
# @brief Check a snpe-deeplab RGBA frame against its input label map.
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date 22 Nov 2026
# @bug No known bugs
#
# The decoder picks a random color per label unless built for NEON, so the
# frame is checked against the labels instead of a golden frame: background
# and labels past the color map are transparent black, every other label is
# opaque and has one color of its own.

import struct
import sys


##
# @brief Return a list of error messages, empty if the frame matches the labels.
def check(labels, pixels, num_labels):
    errors = []
    colors = {}

    if len(labels) != len(pixels):
        return ["%d labels but %d pixels" % (len(labels), len(pixels))]

    for idx, (value, pixel) in enumerate(zip(labels, pixels)):
        label = int(value)
        # The color map has num_labels + 1 entries, background included
        if label == 0 or label > num_labels:
            if pixel != 0:
                errors.append("pixel %d of label %s is 0x%08x, not 0" %
                              (idx, value, pixel))
        elif pixel >> 24 != 0xFF or pixel & 0xFFFFFF == 0:
            errors.append("pixel %d of label %s is 0x%08x, not an opaque color" %
                          (idx, value, pixel))
        elif colors.setdefault(label, pixel) != pixel:
            errors.append("pixel %d of label %d is 0x%08x, not 0x%08x" %
                          (idx, label, pixel, colors[label]))
        if len(errors) >= 10:
            break

    # Random colors of the four labels in the input collide about once in 2.6M runs
    if not errors and len(set(colors.values())) != len(colors):
        errors.append("labels share a color: %s" % colors)

    return errors


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s <label map> <RGBA frame> <number of labels>" % sys.argv[0])
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        data = f.read()
        labels = struct.unpack("<%df" % (len(data) // 4), data)
    with open(sys.argv[2], "rb") as f:
        data = f.read()
        pixels = struct.unpack("<%dI" % (len(data) // 4), data)

    errors = check(labels, pixels, int(sys.argv[3]))
    for error in errors:
        print(error)
    sys.exit(1 if errors else 0)
