#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file checkResult.py
# @brief Check if the tranform results are correct
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date 11 Jul 2018
# @bug No known bugs

import sys

##
# @brief Check dimchg mode with option 0:b, b>0.
#
def testDimchgFirstDimGoHigher(dataA, dataB, dim1, repeatblocksize, elementsize):
  if (len(dataA) != len(dataB)):
    return 1
  loop = len(dataA) / repeatblocksize
  if ((loop * repeatblocksize) != len(dataA)):
    return 2
  ncpy = repeatblocksize / dim1 / elementsize
  if ((ncpy * dim1 * elementsize) != repeatblocksize):
    return 3
  for x in range(0, loop):
    for y in range(0, dim1):
      for z in range(0, ncpy):
        b = x * repeatblocksize + y * ncpy * elementsize + z * elementsize
        a = x * repeatblocksize + z * dim1 * elementsize + y
        if dataA[a:a+elementsize] != dataB[b:b+elementsize]:
          return 4

def readfile (filename):
  F = open(filename, 'r')
  readfile = F.read()
  F.close
  return readfile

if len(sys.argv) < 2:
  exit(5)

if (sys.argv[1] == 'dimchg0:b'):
  if len(sys.argv) < 7:
    exit(5)
  fna = readfile(sys.argv[2])
  fnb = readfile(sys.argv[3])
  dim1 = int(sys.argv[4])
  rbs = int(sys.argv[5])
  es = int(sys.argv[6])
  exit(testDimchgFirstDimGoHigher(fna, fnb, dim1, rbs, es))

exit(5)
