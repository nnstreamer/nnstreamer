#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file checkLabel.py
# @brief Check the result label of caffe2 model
# @author HyoungJoo Ahn <hello.ahn@samsung.com>

import sys
import struct
import string

def readbyte (filename):
  F = open(filename, 'r')
  readbyte = F.read()
  F.close()
  return readbyte

bytearr = readbyte(sys.argv[1])
softmax = []
for i in xrange(10):
  byte = bytearr[i * 4] + bytearr[i * 4 + 1] + bytearr[i * 4 + 2] + bytearr[i * 4 + 3]
  softmax.append(struct.unpack('f', byte))

pred = softmax.index(max(softmax))
answer = string.atoi(sys.argv[2].strip())

exit(pred != answer)
