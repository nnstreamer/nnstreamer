#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2024 Samsung Electronics
#
# @file generateTest.py
# @brief Generate golden test results for padding test cases
# @author Yelin Jeong <yelini.jeong@samsung.com>

import numpy as np


def save_test_data(filename, dtype, cmin, cmax, size, pad):
    data = np.random.randint(cmin, cmax, size=size).astype(dtype)
    with open(filename, 'wb') as file:
        file.write(data.tobytes())
    padded_data = np.pad(data, pad)
    with open(filename + '.golden', 'wb') as file:
        file.write(padded_data.tobytes())


save_test_data('test_00.dat', np.int8, -50, 50, [4, 2, 3, 50, 100], ((0,0),(0,0),(1,1),(0,0),(0,0)))
save_test_data('test_01.dat', np.int8, -50, 50, [2, 3, 50, 100], ((0,0),(0,0),(2,2),(1,1)))
save_test_data('test_02.dat', np.float32, -50, 50, [3, 50, 100], ((1,1),(1,1),(1,1)))

