#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2022 Samsung Electronics
#
# @file generateTest.py
# @brief generate test data and golden test result
# @author Yelin Jeong <yelini.jeong@samsung.com>

import numpy as np

def save_test_data(filename, shape, type):
    data = np.random.uniform(-100, 100, shape).astype(type)
    with open(filename, 'wb') as file:
        file.write(data.tobytes())

    golden = data*2
    with open(filename + '.golden', 'wb') as file:
        file.write(golden.tobytes())

save_test_data('test_00.dat', [4,4,4,4,4], np.float32)
save_test_data('test_01.dat', [3,10,10,4,5,6], np.int8)
