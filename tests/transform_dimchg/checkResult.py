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

import numpy as np

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


##
# @brief Check dimchg mode with option FROM:TO against numpy.
# @param dims The input dimension, innermost first as in tensor_converter.
#
def test_dimchg_any(dataa, datab, dims, dtype, dim_from, dim_to):
    if len(dataa) == 0 or len(dataa) != len(datab):
        return 1
    src = np.frombuffer(dataa, dtype=dtype)
    if src.size % np.prod(dims) != 0:
        return 2
    # numpy is outermost first: tensor dim i is axis (rank - i), after a frame axis
    src = src.reshape([-1] + dims[::-1])
    rank = len(dims)
    expected = np.moveaxis(src, rank - dim_from, rank - dim_to)
    if expected.tobytes() != datab:
        return 4
    return 0


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

if sys.argv[1] == 'dimchg':
    if len(sys.argv) < 8:
        exit(5)
    fna = read_file(sys.argv[2])
    fnb = read_file(sys.argv[3])
    dims = [int(d) for d in sys.argv[4].split(':')]
    dtype = np.dtype(sys.argv[5])
    exit(test_dimchg_any(fna, fnb, dims, dtype, int(sys.argv[6]), int(sys.argv[7])))

exit(5)
