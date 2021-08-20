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
# @brief Check typecast from typea to typeb with file fna/fnb
#
def test_arithmetic(fna, fnb, typeasize, typebsize, typeapack, typebpack, mode, value1, value2):
    lena = len(fna)
    lenb = len(fnb)

    if (0 < (lena % typeasize)) or (0 < (lenb % typebsize)):
        return 10
    num = lena // typeasize
    if num != (lenb // typebsize):
        return 11
    val1 = get_value(value1, typeapack)
    val2 = get_value(value2, typebpack)
    if val1 is None or val2 is None:
        return 12

    for x in range(0, num):
        vala = struct.unpack(typeapack, fna[x * typeasize: x * typeasize + typeasize])[0]
        valb = struct.unpack(typebpack, fnb[x * typebsize: x * typebsize + typebsize])[0]
        if mode == 'add':
            diff = vala + val1 - valb
            if diff > 0.01 or diff < -0.01:
                return 20
        elif mode == 'mul':
            diff = vala * val1 - valb
            if diff > 0.01 or diff < -0.01:
                return 20
        elif mode == 'add-mul':
            diff = (vala + val1) * val2 - valb
            if diff > 0.01 or diff < -0.01:
                return 20
        elif mode == 'mul-add':
            diff = (vala * val1) + val2 - valb
            if diff > 0.01 or diff < -0.01:
                return 20
        else:
            return 21
    return 0


if len(sys.argv) < 2:
    exit(5)

if sys.argv[1] == 'arithmetic':
    if len(sys.argv) < 11:
        exit(5)
    fna = read_file(sys.argv[2])
    fnb = read_file(sys.argv[3])
    typeasize = int(sys.argv[4])
    typebsize = int(sys.argv[5])
    typeapack = (sys.argv[6])
    typebpack = (sys.argv[7])
    mode = sys.argv[8]
    value1 = sys.argv[9]
    value2 = sys.argv[10]

    exit(test_arithmetic(fna, fnb, typeasize, typebsize, typeapack, typebpack, mode, value1, value2))

exit(5)
