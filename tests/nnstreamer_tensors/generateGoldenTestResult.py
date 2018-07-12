#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file generateGoldenTestResult.py
# @brief Generate golden test results for test cases
# @author MyungJoo Ham <myungjoo.ham@samsung.com>

from __future__ import print_function

from struct import *
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gen24bBMP as bmp

##
# @brief Generate Golden Test Case 02, a randomly generated BMP image
# @return (string, string_size, expected_size)
#
def genCase01_BMP_random(colorType, width, height):
    string = ""
    string_size = 0
    sizePerPixel = 3
    if (colorType == 'BGRx'):
        sizePerPixel = 4
    expected_size = width * height * sizePerPixel

    # The result has no stride for other/tensor types.

    if (colorType == 'BGRx'):
        for y in range(0, height):
            for x in range(0, width):
                pval = (random.randrange(256), random.randrange(256), random.randrange(256))
                pixel = pack('BBBB', pval[2], pval[1], pval[0], 255)
                string += pixel
                string_size += 4
    else:
        # Assume RGB
        for y in range(0, height):
            for x in range(0, width):
                pval = (random.randrange(256), random.randrange(256), random.randrange(256))
                pixel = pack('BBB', pval[0], pval[1], pval[2])
                string += pixel
                string_size += 3

    bmp.saveBMP('testcase01_'+colorType+'_'+str(width)+'x'+str(height)+'.bmp', string, colorType, width, height)
    return (string, string_size, expected_size)

def write(filename, string):
    newfile = open(filename, 'wb')
    newfile.write(string)

write('testcase01_RGB_640x480.golden', genCase01_BMP_random('RGB', 640, 480)[0])
