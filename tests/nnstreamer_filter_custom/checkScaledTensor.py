#!/usr/bin/env python

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
#
# @file checkScaledTensor.py
# @brief Check if the scaled results are correct
# @author MyungJoo Ham <myungjoo.ham@samsung.com>

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_utils import read_file


def compare(data1, width1, height1, data2, width2, height2, innerdim):
    if (len(data1) * width2 * height2) != (len(data2) * width1 * height1):
        print(str(len(data1) * width2 * height2) + ' / ' + str(len(data2) * width1 * height1))
        return 1

    count = 0
    count2 = 0
    while count < len(data1):
        # Terminated incorrectly
        if (count + (innerdim * width1 * height1)) > len(data1):
            return 2
        if (count2 + (innerdim * width2 * height2)) > len(data2):
            return 3
        if count2 >= len(data2):
            return 4

        for y in range(0, height2):
            for x in range(0, width2):
                for c in range(0, innerdim):
                    ix = x * width1 // width2
                    iy = y * height1 // height2
                    if data1[count + c + ix * innerdim + iy * width1 * innerdim] != \
                            data2[count2 + c + x * innerdim + y * width2 * innerdim]:
                        print('At ' + str(x) + ',' + str(y))
                        return 5
        count = count + innerdim * width1 * height1
        count2 = count2 + innerdim * width2 * height2

    if count > len(data1):
        return 6
    if count2 > len(data2):
        return 7
    return 0


if len(sys.argv) != 8:
    exit(9)

data1 = read_file(sys.argv[1])
width1 = int(sys.argv[2])
height1 = int(sys.argv[3])
data2 = read_file(sys.argv[4])
width2 = int(sys.argv[5])
height2 = int(sys.argv[6])
innerdim = int(sys.argv[7])

exit(compare(data1, width1, height1, data2, width2, height2, innerdim))
