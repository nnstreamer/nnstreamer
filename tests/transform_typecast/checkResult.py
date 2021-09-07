#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
#
# @file checkResult.py
# @brief Check if the transform results are correct with typecast
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date 20 Jul 2018
# @bug No known bugs

import os
import sys
import struct

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_utils import read_file


def compare_float(a, b):
    diff = float(a) - float(b)
    return not (diff > 0.01 or diff < -0.01)


def compare_int(a, b, maskb):
    vala = int(a) & maskb
    valb = int(b)
    # Remove the signedness characteristics!
    diff = (vala ^ valb) & maskb
    return not (diff != 0)


##
# @brief Auxiliary function to check args of test_typecast
#
def _check_args(lena, typeasize, lenb, typebsize):
    if (0 < (lena % typeasize)) or (0 < (lenb % typebsize)):
        return False
    if (lena // typeasize) != (lenb // typebsize):
        return False
    return True


##
# @brief Auxiliary function to compare values
#
def _compare(vala, valb, typeb, maskb):
    if typeb[0:5] == 'float':
        if not compare_float(vala, valb):
            return False
    elif typeb[0:4] == 'uint' or typeb[0:3] == 'int':
        if not compare_int(vala, valb, maskb):
            return False
    else:
        return False

    return True


##
# @brief Check typecast from typea to typeb with file fna/fnb
#
def test_typecast(d1, d2):
    # data should be tuple (data, type, type-size, type-pack)
    fna, typea, typeasize, typeapack = d1
    fnb, typeb, typebsize, typebpack = d2
    lena = len(fna)
    lenb = len(fnb)

    if not _check_args(lena, typeasize, lenb, typebsize):
        return 10

    num = lena // typeasize
    limitb = 2 ** (8 * typebsize)
    maskb = limitb - 1

    for x in range(0, num):
        astart = x * typeasize
        bstart = x * typebsize
        vala = struct.unpack(typeapack, fna[astart: astart + typeasize])[0]
        valb = struct.unpack(typebpack, fnb[bstart: bstart + typebsize])[0]
        if not _compare(vala, valb, typeb, maskb):
            return 21
    return 0


if len(sys.argv) < 2:
    exit(5)

if sys.argv[1] == 'typecast':
    if len(sys.argv) < 10:
        exit(5)

    # (data, type, type-size, type-pack)
    data1 = (read_file(sys.argv[2]), sys.argv[4], int(sys.argv[5]), sys.argv[6])
    data2 = (read_file(sys.argv[3]), sys.argv[7], int(sys.argv[8]), sys.argv[9])

    exit(test_typecast(data1, data2))

exit(5)
