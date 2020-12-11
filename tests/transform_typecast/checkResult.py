#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
#
# @file checkResult.py
# @brief Check if the tranform results are correct with typecast
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date 20 Jul 2018
# @bug No known bugs

import sys
import struct

##
# @brief Check typecast from typea to typeb with file fna/fnb
#
def testTypeCast (fna, fnb, typea, typeasize, typeapack, typeb, typebsize, typebpack):
  lena = len(fna)
  lenb = len(fnb)

  if (0 < (lena % typeasize)) or (0 < (lenb % typebsize)):
    return 10
  num = lena // typeasize
  if num != (lenb // typebsize):
    return 11
  limitb = 2 ** (8 * typebsize)
  maskb = limitb - 1

  for x in range(0, num):
    vala = struct.unpack(typeapack, fna[x * typeasize: x * typeasize + typeasize])[0]
    valb = struct.unpack(typebpack, fnb[x * typebsize: x * typebsize + typebsize])[0]
    if (typeb[0:5] == 'float'):
      diff = float(vala) - valb
      if diff > 0.01 or diff < -0.01:
        return 20
    elif (typeb[0:4] == 'uint' or typeb[0:3] == 'int'):
      vala = int(vala)
      vala = vala & maskb
      valb = int(valb)

      # Remove the signedness characteristics!
      diff = (vala ^ valb) & maskb
      if (diff != 0):
        print ("WTH?? "+str(vala)+"->"+str(valb)+"\n")
        return 20
    else:
      return 21
  return 0

def readfile (filename):
  F = open(filename, 'rb')
  readfile = F.read()
  F.close
  return readfile

if len(sys.argv) < 2:
  exit(5)

if (sys.argv[1] == 'typecast'):
  if len(sys.argv) < 10:
    exit(5)
  fna = readfile(sys.argv[2])
  fnb = readfile(sys.argv[3])
  typea = sys.argv[4]
  typeasize = int(sys.argv[5])
  typeapack = sys.argv[6]
  typeb = sys.argv[7]
  typebsize = int(sys.argv[8])
  typebpack = sys.argv[9]
  exit(testTypeCast(fna, fnb, typea, typeasize, typeapack, typeb, typebsize, typebpack))

exit(5)
