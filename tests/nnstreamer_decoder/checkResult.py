#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2022 Samsung Electronics
#
# @file checkResult.py
# @brief Check if the tensor decoder results is correct with the padded video.
# @author Gichan Jang <gichan2.jang@samsung.com>
# @date 3 Jan 2022
# @bug No known bugs

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_utils import compare_video_frame


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Wrong # of parameters")
        exit(-1)

    golden = sys.argv[1]
    raw = sys.argv[2]
    colorspace = sys.argv[3]
    width = int(sys.argv[4])
    height = int(sys.argv[5])

    # (godlden file, raw file, colorspace, width, height)
    exit(compare_video_frame(golden, raw, colorspace, width, height))
