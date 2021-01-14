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
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from struct import pack
import random

import numpy as np

def saveTestData(filename, width, height):
    data = []

    for w in range(0,width):
        for h in range(0,height):
            n = random.uniform(0.0, 10.0)
            data.append(n)

    string = pack('%df' % (len(data)), *data)
    with open(filename,'wb') as file:
        file.write(string)

    file.close()

    a=np.array(data)
    mean = np.mean(a)
    standard = np.std(a)
    result=abs((a-np.mean(a)) / (np.std(a)+1e-10))

    data = []
    for w in range(0,width):
        for h in range(0,height):
            data.append(result[w*height + h])

    string = pack('%df' % (len(data)), *data)
    with open(filename+".golden",'wb') as file1:
        file1.write(string)

    return result, mean, standard

buf = saveTestData("test_00.dat", 100, 50)
