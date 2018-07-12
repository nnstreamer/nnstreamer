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
# @brief Generate Golden Test Case 01, a single videosrctest frame of 280x40xRGB
# @return (string, string_size, expected_size)
#
# If string_size < expected_size, you do not need to check the results offset >= string_size.
# string: binary string (b'\xff\x00....')
def genCase01_RGB():
    string = ""
    string_size = 0
    expected_size = 280 * 40 * 3
    for i in range (0, 26):
        # White
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 255, 255, 255)
        # Yellow
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 255, 255, 0)
        # Light Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 255, 255)
        # Green
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 255, 0)
        # Purple
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 255, 0, 255)
        # Red
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 255, 0, 0)
        # Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 0, 255)
    for i in range (26, 30):
        # Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 0, 0)
        # Purple
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 255, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 0, 0)
        # Light Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 255, 255)
        # Black
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 0, 0, 0)
        # White
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBB', 255, 255, 255)
    for i in range (0, 46):
        # Dark Blue
        string_size = string_size + 46
        string += pack('BBB', 0, 0, 128)
    for i in range (46, 93):
        # White
        string_size = string_size + 47
        string += pack('BBB', 255, 255, 255)
    for i in range (93, 140):
        # Gray Blue
        string_size = string_size + 47
        string += pack('BBB', 0, 128, 255)
    for i in range (140, 186):
        # Black
        string_size = string_size + 46
        string += pack('BBB', 0, 0, 0)
    for i in range (186, 210):
        # Dark Gray
        string_size = string_size + 24
        string += pack('BBB', 19, 19, 19)
    # We do not check the reset pixels: they are randomly generated.
    string_size = string_size * 3

    return (string, string_size, expected_size)

##
# @brief Generate Golden Test Case 01, a single videosrctest frame of 280x40xBGRx
# @return (string, string_size, expected_size)
#
def genCase01_BGRx():
    string = ""
    string_size = 0
    expected_size = 280 * 40 * 4
    for i in range (0, 26):
        # White
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 255, 255, 255)
        # Yellow
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 0, 255, 255, 255)
        # Light Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 255, 0, 255)
        # Green
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 0, 255, 0, 255)
        # Purple
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 0, 255, 255)
        # Red
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 0, 0, 255, 255)
        # Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 0, 0, 255)
    for i in range (26, 30):
        # Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 0, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 0, 0, 0, 255)
        # Purple
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 0, 255, 255)
        # Black
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 0, 0, 0, 255)
        # Light Blue
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 255, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 0, 0, 0, 255)
        # White
        string_size = string_size + 40
        for j in range (0, 40):
            string += pack('BBBB', 255, 255, 255, 255)
    for i in range (0, 46):
        # Dark Blue
        string_size = string_size + 46
        string += pack('BBBB', 128, 0, 0, 255)
    for i in range (46, 93):
        # White
        string_size = string_size + 47
        string += pack('BBBB', 255, 255, 255, 255)
    for i in range (93, 140):
        # Gray Blue
        string_size = string_size + 47
        string += pack('BBBB', 255, 128, 0, 255)
    for i in range (140, 186):
        # Black
        string_size = string_size + 46
        string += pack('BBBB', 0, 0, 0, 255)
    for i in range (186, 210):
        # Dark Gray
        string_size = string_size + 24
        string += pack('BBBB', 19, 19, 19, 255)
    # We do not check the reset pixels: they are randomly generated.
    string_size = string_size * 4
    return (string, string_size, expected_size)

##
# @brief Generate Golden Test Case 02, a randomly generated BMP image
# @return (string, string_size, expected_size)
#
def genCase02_BMP_random(colorType, width, height):
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

    bmp.saveBMP("testcase02_"+colorType+'_'+str(width)+'x'+str(height)+".bmp", string, colorType, width, height)
    return (string, string_size, expected_size)

##
# @brief Generate a fixed BMP sequence for stream test
# @return 0 if success. non-zero if failed.
#
# This gives "16x16", black, white, green, red, blue, wb-checker, rb-checker, gr-checker, red-cross-on-white, blue-cross-on-black (4x4 x 16, left-top/right-bottom white/red/green). "10 files" with 0 ~ 9 postfix in the filename
def genCase08_BMP_stream(filename_prefix, goldenfilename):
    string = ["", "", "", "", "", "", "", "", "", ""]
    sizePerPixel = 3
    sizex = 16
    sizey = 16

    for y in range(0, sizey):
        for x in range(0, sizex):
            # black. Frame 0
            string[0] += pack('BBB', 0, 0, 0)
            # white. Frame 1
            string[1] += pack('BBB', 255, 255, 255)
            # green, Frame 2
            string[2] += pack('BBB', 0, 255, 0)
            # red, Frame 3
            string[3] += pack('BBB', 255, 0, 0)
            # blue, Frame 4
            string[4] += pack('BBB', 0, 0, 255)
            # white-black checker, Frame 5
            if (((x / 4) % 2) + ((y / 4) % 2)) == 1:
                string[5] += pack('BBB', 0, 0, 0)
            else:
                string[5] += pack('BBB', 255, 255, 255)
            # red-blue checker, Frame 6
            if (((x / 4) % 2) + ((y / 4) % 2)) == 1:
                string[6] += pack('BBB', 0, 0, 255)
            else:
                string[6] += pack('BBB', 255, 0, 0)
            # green-red checker, Frame 7
            if (((x / 4) % 2) + ((y / 4) % 2)) == 1:
                string[7] += pack('BBB', 255, 0, 0)
            else:
                string[7] += pack('BBB', 0, 255, 0)
            # red-cross-on-white, Frame 8
            if x == y:
                string[8] += pack('BBB', 255, 0, 0)
            else:
                string[8] += pack('BBB', 255, 255, 255)
            # blue-cross-on-black, Frame 9
            if x == y:
                string[9] += pack('BBB', 0, 0, 255)
            else:
                string[9] += pack('BBB', 0, 0, 0)

    newfile = open(goldenfilename, 'wb')
    for i in range(0, 10):
        newfile.write(string[i])
        bmp.saveBMP(filename_prefix + str(i) + '.bmp', string[i], 'RGB', 16, 16)

##
# @brief Write the generated data
#
def write(filename, string):
    newfile = open(filename, 'wb')
    newfile.write(string)

# Allow to create specific cases only if proper argument is given
target=-1 # -1 == ALL
if len(sys.argv) > 2: # There's some arguments
    target = int(sys.argv[1])

if target == -1 or target == 1:
    write('testcase01.rgb.golden', genCase01_RGB()[0])
    write('testcase01.bgrx.golden', genCase01_BGRx()[0])
if target == -1 or target == 2:
    write('testcase02_RGB_640x480.golden', genCase02_BMP_random('RGB', 640, 480)[0])
    write('testcase02_BGRx_640x480.golden', genCase02_BMP_random('BGRx', 640, 480)[0])
    write('testcase02_RGB_642x480.golden', genCase02_BMP_random('RGB', 642, 480)[0])
    write('testcase02_BGRx_642x480.golden', genCase02_BMP_random('BGRx', 642, 480)[0])
if target == -1 or target == 8:
    genCase08_BMP_stream('testsequence_', 'testcase08.golden')
if target == -1 or target == 9:
    str = genCase02_BMP_random('RGB', 100, 100)[0]
    write('testcase01_RGB_100x100.golden', str)
    write('testcase02_RGB_100x100.golden', str+str)
    write('testcase03_RGB_100x100.golden', str+str+str)
