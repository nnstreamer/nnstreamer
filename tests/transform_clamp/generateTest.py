#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2020 Samsung Electronics
#
# @file generateTest.py
# @brief Generate golden test results for clamp test cases
# @author Dongju Chae <dongju.chae@samsung.com>

import sys
import os
import numpy as np


def save_test_data(filename, dtype, cmin, cmax):
    data = np.random.randint(-100, 100, size=[100, 50]).astype(dtype)
    with open(filename, 'wb') as file:
        file.write(data.tobytes())

    np.clip(data, cmin, cmax, out=data)
    with open(filename + '.golden', 'wb') as file:
        file.write(data.tobytes())


save_test_data('test_00.dat', np.int8, -50, 50)
save_test_data('test_01.dat', np.uint32, 20, 80)
save_test_data('test_02.dat', np.float32, -33.3, 77.7)
