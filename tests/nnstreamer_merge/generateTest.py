#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
#
# @file generateTest.py
# @brief Generate golden test results for test cases
# @author Jijoong Moon <jijoong.moon@samsung.com>

import sys
import os
import random
import numpy as np
from struct import pack


def save_test_data(filename, width, height, channel, batch):
    data = []

    for b in range(0, batch):
        for h in range(0, height):
            for w in range(0, width):
                for c in range(0, channel):
                    data.append(random.uniform(0.0, 10.0))

    string = pack('%df' % (len(data)), *data)
    with open(filename, 'wb') as fd:
        fd.write(string)

    return data


# merge with channel direction
ch = [3, 2, 4]
width = 100
height = 50
batch = 1

buf = []
buf.append(save_test_data("channel_00.dat", width, height, 3, batch))
buf.append(save_test_data("channel_01.dat", width, height, 2, batch))
buf.append(save_test_data("channel_02.dat", width, height, 4, batch))

out_data = []
for b in range(0, batch):
    for h in range(0, height):
        for w in range(0, width):
            for n in range(0, 3):
                for c in range(0, ch[n]):
                    out_data.append(buf[n][b * height * width * ch[n] + h * width * ch[n] + w * ch[n] + c])

out = pack('%df' % (len(out_data)), *out_data)
with open("channel.golden", 'wb') as file:
    file.write(out)

# merge with width direction
width = [100, 200, 300]
ch = 3
height = 50
batch = 1

buf = []
buf.append(save_test_data("width_100.dat", width[0], height, ch, batch))
buf.append(save_test_data("width_200.dat", width[1], height, ch, batch))
buf.append(save_test_data("width_300.dat", width[2], height, ch, batch))

out_data = []
for b in range(0, batch):
    for h in range(0, height):
        for n in range(0, 3):
            for w in range(0, width[n]):
                for c in range(0, ch):
                    out_data.append(buf[n][b * height * width[n] * ch + h * width[n] * ch + w * ch + c])

out = pack('%df' % (len(out_data)), *out_data)
with open("width.golden", 'wb') as file:
    file.write(out)

# merge with width direction
batch = [1, 2, 3]
ch = 3
height = 50
width = 100

buf = []
buf.append(save_test_data("batch_1.dat", width, height, ch, batch[0]))
buf.append(save_test_data("batch_2.dat", width, height, ch, batch[1]))
buf.append(save_test_data("batch_3.dat", width, height, ch, batch[2]))

out_data = []
for n in range(0, 3):
    for b in range(0, batch[n]):
        for h in range(0, height):
            for w in range(0, width):
                for c in range(0, ch):
                    out_data.append(buf[n][b * height * width * ch + h * width * ch + w * ch + c])

out = pack('%df' % (len(out_data)), *out_data)
with open("batch.golden", 'wb') as file:
    file.write(out)
