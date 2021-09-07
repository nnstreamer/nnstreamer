#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file checkResult.py
# @brief Check if the transform results are correct with typecast
# @author Jijoong Moon <jijoong.moo@samsung.com>
# @date 20 Jul 2018
# @bug No known bugs

import os
import sys
import struct

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_utils import read_file


##
# @brief Get value of proper type corresponding to the unpack format string
#
def get_value(val, unpack_format):
    if unpack_format in ['b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q']:
        return int(val)
    elif unpack_format in ['f', 'd']:
        return float(val)
    else:
        return None

##
# @brief Auxiliary function to check args of test_arithmetic
#
def _check_args(lena, typeasize, lenb, typebsize):
    if (0 < (lena % typeasize)) or (0 < (lenb % typebsize)):
        return False
    if (lena // typeasize) != (lenb // typebsize):
        return False
    return True


##
# @brief Auxiliary function to check difference
#
def _check_diff(mode, va, vb, val1, val2):
    if mode == 'add':
        ret = va + val1
    elif mode == 'mul':
        ret = va * val1
    elif mode == 'add-mul':
        ret = (va + val1) * val2
    elif mode == 'mul-add':
        ret = (va * val1) + val2
    else:
        return False

    diff = ret - vb
    if abs(diff) > 0.01:
        return False
    else:
        return True


##
# @brief Check typecast from typea to typeb with file fna/fnb
#
def test_arithmetic(d1, d2, mode, v1, v2):
    # data should be tuple (data, type-size, type-pack)
    # value should be tuple (value, type-pack)
    fna, typeasize, typeapack = d1
    fnb, typebsize, typebpack = d2
    lena = len(fna)
    lenb = len(fnb)

    if not _check_args(lena, typeasize, lenb, typebsize):
        return 10

    num = lena // typeasize
    val1 = get_value(v1[0], v1[1])
    val2 = get_value(v2[0], v2[1])
    if val1 is None or val2 is None:
        return 12

    for x in range(0, num):
        astart = x * typeasize
        bstart = x * typebsize
        vala = struct.unpack(typeapack, fna[astart: astart + typeasize])[0]
        valb = struct.unpack(typebpack, fnb[bstart: bstart + typebsize])[0]
        if not _check_diff(mode, vala, valb, val1, val2):
            return 20
    return 0


if len(sys.argv) < 2:
    exit(5)

if sys.argv[1] == 'arithmetic':
    if len(sys.argv) < 11:
        exit(5)

    packa = (sys.argv[6])
    packb = (sys.argv[7])
    arith_mode = sys.argv[8]

    # (data, type-size, type-pack)
    data1 = (read_file(sys.argv[2]), int(sys.argv[4]), packa)
    data2 = (read_file(sys.argv[3]), int(sys.argv[5]), packb)
    # (value, type-pack)
    value1 = (sys.argv[9], packa)
    value2 = (sys.argv[10], packb)

    exit(test_arithmetic(data1, data2, arith_mode, value1, value2))

exit(5)
