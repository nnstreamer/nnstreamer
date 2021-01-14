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

def saveTestData(filename, channel, height, width, batch, idx_i, idx_j, idx_k, idx_l):
    data = []

    for b in range(0,batch):
        for c in range(0,channel):
            for h in range(0,height):
                for w in range(0,width):
                    n = random.uniform(0.0, 10.0)
                    data.append(n)

    string = pack('%df' % (len(data)), *data)
    with open(filename,'wb') as file:
        file.write(string)

    file.close()

    a=np.ndarray((batch, channel, height, width), np.float32);

    for b in range(0,batch):
        for c in range(0,channel):
            for h in range(0,height):
                for w in range(0,width):
                    a[b][c][h][w] = data[b*channel*height*width+c*height*width+h*width+w]

    a=np.transpose(a, (idx_i,idx_j,idx_k,idx_l))

    a=a.copy(order='C')

    with open(filename+".golden",'wb') as file1:
        file1.write(a)
    file1.close()

    return a

buf = saveTestData("test01_00.dat", 3, 50, 100, 1, 0, 2, 3, 1)

buf = saveTestData("test02_00.dat", 3, 100, 200, 1, 0, 2, 3, 1)

buf = saveTestData("test03_00.dat", 3, 100, 200, 1, 0, 1, 3, 2)
