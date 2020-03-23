#!/usr/bin/env python

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
    string = b''
    data = []

    for w in range(0,width):
        for h in range(0,height):
            n = random.uniform(0.0, 10.0)
            string += pack('f', n)
            data.append(n)

    with open(filename,'wb') as file:
        file.write(string)

    file.close()

    a=np.array(data)
    mean = np.mean(a)
    standard = np.std(a)
    result=abs((a-np.mean(a)) / (np.std(a)+1e-10))

    s = b''
    for w in range(0,width):
        for h in range(0,height):
            s += pack('f',result[w*height+h])

    with open(filename+".golden",'wb') as file1:
        file1.write(s)

    return result, mean, standard

buf = saveTestData("test_00.dat", 100, 50)
