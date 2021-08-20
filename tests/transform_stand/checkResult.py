#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
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
# @brief Check typecast from typea to typeb with file fna/fnb
#
def test_standardization(fna, fnb, typeasize, typebsize, typeapack, typebpack):
    lena = len(fna)
    lenb = len(fnb)

    if (0 < (lena % typeasize)) or (0 < (lenb % typebsize)):
        return 10
    num = lena // typeasize
    if num != (lenb // typebsize):
        return 11

    for x in range(0, num):
        vala = struct.unpack(typeapack, fna[x * typeasize: x * typeasize + typeasize])[0]
        valb = struct.unpack(typebpack, fnb[x * typebsize: x * typebsize + typebsize])[0]
        diff = vala - valb
        if diff > 0.00001 or diff < -0.00001:
            return 20
    return 0


if len(sys.argv) < 2:
    exit(5)

if sys.argv[1] == 'standardization':
    if len(sys.argv) < 8:
        exit(5)
    fna = read_file(sys.argv[2])
    fnb = read_file(sys.argv[3])
    typeasize = int(sys.argv[4])
    typebsize = int(sys.argv[5])
    typeapack = (sys.argv[6])
    typebpack = (sys.argv[7])

    exit(test_standardization(fna, fnb, typeasize, typebsize, typeapack, typebpack))

exit(5)
