#!/usr/bin/env python

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2019 Samsung Electronics
#
# @file checkLabel.py
# @brief Check the result label of pytorch model
# @author Parichay Kapoor <pk.kapoor@samsung.com>

import sys
import os
import struct
import string

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gen24bBMP import convert_to_bytes


# Read the bytes from the file
def read_bytes(filename):
    with open(filename, 'rb') as f:
        b = f.read()
    return b


# Verify that the output of test case verifies the filename of the input
onehot = read_bytes(sys.argv[1])
onehot = [convert_to_bytes(x) for x in onehot]
idx = onehot.index(max(onehot))

label = str(idx)

answer = sys.argv[2].split('/')[-1].split('.')[0].strip()
exit(label != answer)
