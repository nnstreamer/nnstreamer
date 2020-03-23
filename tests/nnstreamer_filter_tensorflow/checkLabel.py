#!/usr/bin/env python

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
from gen24bBMP  import convert_to_bytes

if sys.version_info < (3,):
    range = xrange

def readbyte (filename):
    F = open(filename, 'rb')
    readbyte = F.read()
    F.close()
    return readbyte


def readlabel (filename):
    F = open(filename, 'r')
    line = F.readlines()
    F.close()
    return line

bytearr = readbyte(sys.argv[1])
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
