#!/usr/bin/env python

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2020 Samsung Electronics
#
# @file CheckMnist.py
# @brief Check the result label of MNIST model
# @author Sangjung Woo <sangjung.woo@samsung.com>

import struct
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gen24bBMP  import convert_to_bytes

def get_label(fname):
	f = open(fname, 'rb')
	rbyte = f.read()
	f.close()
	
	score_list = []
	for i in range(10):
		byte = b''
		byte += convert_to_bytes(rbyte[i * 4])
		byte += convert_to_bytes(rbyte[i * 4 + 1])
		byte += convert_to_bytes(rbyte[i * 4 + 2])
		byte += convert_to_bytes(rbyte[i * 4 + 3])
		score_list.append(struct.unpack('f', byte))

	return score_list.index(max(score_list))


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Wrong # of parameters")
		exit (-1)

	retfile = sys.argv[1]
	answer = int(sys.argv[2].strip())

	exit(get_label(retfile) != answer)
