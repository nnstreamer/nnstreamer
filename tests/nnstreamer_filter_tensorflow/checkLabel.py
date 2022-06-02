#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
#
# @file checkLabel.py
# @brief Check the result label of tensorflow-lite model
# @author HyoungJoo Ahn <hello.ahn@samsung.com>

import sys
import os
import struct
import string

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from test_utils import convert_to_bytes, read_file


bytearr = read_file(sys.argv[1])
softmax = []
for i in range(10):
    byte = b''
    byte += convert_to_bytes(bytearr[i * 4])
    byte += convert_to_bytes(bytearr[i * 4 + 1])
    byte += convert_to_bytes(bytearr[i * 4 + 2])
    byte += convert_to_bytes(bytearr[i * 4 + 3])
    softmax.append(struct.unpack('f', byte))

pred = softmax.index(max(softmax))
answer = int(sys.argv[2].strip())
exit(pred != answer)
