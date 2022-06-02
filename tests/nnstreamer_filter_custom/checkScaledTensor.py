#!/usr/bin/env python3

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
from test_utils import read_file, compare_scaled_tensor


if len(sys.argv) != 8:
    exit(9)

# (data, width, height)
data1 = (read_file(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
data2 = (read_file(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
innerdim = int(sys.argv[7])

exit(compare_scaled_tensor(data1, data2, innerdim))
