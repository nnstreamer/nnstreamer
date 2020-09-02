#!/usr/bin/env python

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
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from struct import pack
import numpy as np

def saveTestData(filename, dtype, cmin, cmax):
  data = np.random.randint(-100, 100, size=[100, 50]).astype(dtype)
  with open (filename, 'wb') as file:
    file.write (data)

  np.clip (data, cmin, cmax, out=data)
  with open (filename + ".golden", 'wb') as file:
    file.write (data)

saveTestData("test_00.dat", np.int8, -50, 50)
saveTestData("test_01.dat", np.uint32, 20, 80)
saveTestData("test_02.dat", np.float32, -33.3, 77.7)
