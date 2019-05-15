#!/usr/bin/env python

##
# Copyright (C) 2019 Samsung Electronics
# License: LGPL-2.1
#
# @file checkLabel.py
# @brief Check the result label of pytorch model
# @author Parichay Kapoor <pk.kapoor@samsung.com>

import sys

# Read the bytes from the file
def readbyte (filename):
  with open(filename, 'r') as f:
    readbyte = f.read()
  return readbyte

# Verify that the output of test case verifies the filename of the input
onehot = readbyte(sys.argv[1])
onehot = [str(ord(x)) for x in onehot]
idx = onehot.index(max(onehot))

label = str(idx)

answer = sys.argv[2].split('/')[1].split('.')[0].strip()
exit(label != answer)
