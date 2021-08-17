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
from gen24bBMP import convert_to_bytes


def read_bytes(filename):
    with open(filename, 'rb') as file:
        b = file.read()
    return b


def read_label(filename):
    with open(filename, 'r') as file:
        b = file.readlines()
    return b


onehot = read_bytes(sys.argv[1])
onehot = [convert_to_bytes(x) for x in onehot]
idx = onehot.index(max(onehot))

label_list = read_label(sys.argv[2])
label = label_list[idx].strip()

answer = sys.argv[3].strip()
exit(label != answer)
