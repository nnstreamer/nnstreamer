#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
#
# @file checkResult.py
# @brief Check if the transform results are correct
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date 11 Jul 2018
# @bug No known bugs

import os
import sys
from itertools import product

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_utils import read_file


##
# @brief Check dimchg mode with option 0:b, b>0.
#
def test_dimchg(dataa, datab, dim, repeatblocksize, elementsize):
    if len(dataa) != len(datab):
        return 1
    loop = len(dataa) // repeatblocksize
    if (loop * repeatblocksize) != len(dataa):
        return 2
    ncpy = repeatblocksize // dim // elementsize
    if (ncpy * dim * elementsize) != repeatblocksize:
        return 3
    for x, y, z in product(range(0, loop), range(0, dim), range(0, ncpy)):
        b = x * repeatblocksize + y * ncpy * elementsize + z * elementsize
        a = x * repeatblocksize + z * dim * elementsize + y
        if dataa[a:a + elementsize] != datab[b:b + elementsize]:
            return 4


if len(sys.argv) < 2:
    exit(5)

if sys.argv[1] == 'dimchg0:b':
    if len(sys.argv) < 7:
        exit(5)
    fna = read_file(sys.argv[2])
    fnb = read_file(sys.argv[3])
    dim1 = int(sys.argv[4])
    rbs = int(sys.argv[5])
    es = int(sys.argv[6])
    exit(test_dimchg(fna, fnb, dim1, rbs, es))

exit(5)
