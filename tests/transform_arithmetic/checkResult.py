#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file checkResult.py
# @brief Check if the tranform results are correct with typecast
# @author Jijoong Moon <jijoong.moo@samsung.com>
# @date 20 Jul 2018
# @bug No known bugs

import sys
import struct

##
# @brief Check typecast from typea to typeb with file fna/fnb
#
def testArithmetic (fna, fnb, typeasize, typebsize,typeapack, typebpack, mode, value):
  lena = len(fna)
  lenb = len(fnb)

  if (0 < (lena % typeasize)) or (0 < (lenb % typebsize)):
    return 10
  num = lena / typeasize
  if num != (lenb / typebsize):
    return 11
  limitb = 2 ** (8 * typebsize)
  maskb = limitb - 1
  for x in range(0, num):
    vala = struct.unpack(typeapack, fna[x * typeasize: x * typeasize + typeasize])[0]
    valb = struct.unpack(typebpack, fnb[x * typebsize: x * typebsize + typebsize])[0]
    if (mode[0:3] == 'add'):
      diff = vala + float(value) - valb
      if diff > 0.01 or diff < -0.01:
        return 20
    elif (mode[0:3] == 'mul'):
      diff = vala * float(value) - valb
      if diff > 0.01 or diff < -0.01:
        return 20
    else:
      return 21
  return 0

def readfile (filename):
  F = open(filename, 'r')
  readfile = F.read()
  F.close
  return readfile

if len(sys.argv) < 2:
  exit(5)

if (sys.argv[1] == 'arithmetic'):
  if len(sys.argv) < 10:
    exit(5)
  fna = readfile(sys.argv[2])
  fnb = readfile(sys.argv[3])
  typeasize = int(sys.argv[4])
  typebsize = int(sys.argv[5])
  typeapack = (sys.argv[6])
  typebpack = (sys.argv[7])
  mode = sys.argv[8]
  value = sys.argv[9]

  exit(testArithmetic(fna, fnb, typeasize, typebsize, typeapack, typebpack, mode, value))

exit(5)
