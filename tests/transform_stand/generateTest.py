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
from itertools import product
from struct import pack


def save_test_data(filename, width, height, dc_average=False):
    data = []
    for w, h in product(range(0, width), range(0, height)):
        data.append(random.uniform(0.0, 10.0))

    string = pack('%df' % (len(data)), *data)
    with open(filename, 'wb') as file:
        file.write(string)

    a = np.array(data)
    mean = np.mean(a)
    standard = np.std(a)
    if dc_average:
        result = a - mean
    else:
        result = abs((a - mean) / (standard + 1e-10))

    data = []
    for w, h in product(range(0, width), range(0, height)):
        data.append(result[w * height + h])

    string = pack('%df' % (len(data)), *data)
    with open(filename + '.golden', 'wb') as file1:
        file1.write(string)


def save_test_per_channel_data(filename, num_channel, num_sample, dc_average=False):
    arr = np.random.randn(num_sample, num_channel)
    with open(filename, 'wb') as f:
        f.write(arr.astype('f').tobytes())

    means = np.mean(arr, axis=0)
    result = arr - means

    if not dc_average:
        std = np.std(arr, axis=0)
        result = abs(result / (std + 1e-10))

    with open(filename + '.golden', 'wb') as f:
        f.write(result.astype('f').tobytes())


save_test_data('test_00.dat', 100, 50, False)
save_test_per_channel_data('test_01.dat', 50, 100, False)
save_test_data('test_02.dat', 100, 50, True)
save_test_per_channel_data('test_03.dat', 50, 100, True)
