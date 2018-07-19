#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file generateGoldenTestResult.py
# @brief Generate golden test results for test cases
# @author Jijoong Moon <jijoong.moon@samsung.com>
#

from __future__ import print_function

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gen24bBMP as bmp


bmp.gen_BMP_random('RGB', 640, 480, 'testcase01')
bmp.gen_BMP_random('BGRx', 640, 480, 'testcase01')
bmp.gen_BMP_random('RGB', 642, 480, 'testcase01')
bmp.gen_BMP_random('BGRx', 642, 480, 'testcase01')
