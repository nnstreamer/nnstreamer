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
import numpy as np
import random
from itertools import product
from struct import pack


def save_test_data(filename, channel, height, width, batch, idx_i, idx_j, idx_k, idx_l):
    data = []
    for b, c, h, w in product(range(0, batch), range(0, channel), range(0, height), range(0, width)):
        data.append(random.uniform(0.0, 10.0))

    string = pack('%df' % (len(data)), *data)
    with open(filename, 'wb') as file:
        file.write(string)

    a = np.ndarray((batch, channel, height, width), np.float32)

    for b, c, h, w in product(range(0, batch), range(0, channel), range(0, height), range(0, width)):
        a[b][c][h][w] = data[b * channel * height * width + c * height * width + h * width + w]

    a = np.transpose(a, (idx_i, idx_j, idx_k, idx_l))
    a = a.copy(order='C')

    with open(filename + '.golden', 'wb') as file1:
        file1.write(a)


save_test_data('test01_00.dat', 3, 50, 100, 1, 0, 2, 3, 1)
save_test_data('test02_00.dat', 3, 100, 200, 1, 0, 2, 3, 1)
save_test_data('test03_00.dat', 3, 100, 200, 1, 0, 1, 3, 2)
