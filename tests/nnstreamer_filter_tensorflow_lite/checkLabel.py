#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
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

onehot = readbyte(sys.argv[1])
onehot = [convert_to_bytes(x) for x in onehot]
idx = onehot.index(max(onehot))

label_list = readlabel(sys.argv[2])
label = label_list[idx].strip()

answer = sys.argv[3].strip()
exit(label != answer)
