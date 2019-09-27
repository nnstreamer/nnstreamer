#!/usr/bin/env python

##
# Copyright (C) 2018 Samsung Electronics
# License: LGPL-2.1
#
# @file gen24bBMP.py
# @brief Generate 24bpp .bmp files for test cases
# @author MyungJoo Ham <myungjoo.ham@samsung.com>

from __future__ import print_function

from struct import pack
import random
import sys

##
# @brief Convert given data to bytes
# @param[in] data The data to be converted to bytes array
# @return bytes converted from the data

def convert_to_bytes(data):
    """
    Convert given data to bytes

    @param  data: The data to be converted to bytes
    @rtype      : bytes
    @return     : bytes converted from the data
    """

    if isinstance(data, bytes):
        return data
    else:
        return pack("<B", data)

##
# @brief Save bitmap "data" to "filename"
# @param[in] filename The filename to be saves as a .bmp file.
# @param[in] data string of RGB (packed in 'BBB') or BGRx (packed in 'BBBB')
# @param[in] colorspace "RGB" or "BGRx"
# @param[in] width Width of the picture
# @param[in] height Height of the picture
def saveBMP(filename, data, colorspace, width, height):
    size = len(data)
    graphics = b''
    # Default value of bytes per pixel value for RGB
    bytes_per_px = 3

    if colorspace == 'RGB':
        assert(size == (width * height * bytes_per_px))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = 3 * (w + width * h)
                graphics += convert_to_bytes(data[pos + 2])
                graphics += convert_to_bytes(data[pos + 1])
                graphics += convert_to_bytes(data[pos])
            for x in range(0, (width * 3) % 4):
                graphics += pack('<B', 0)
    elif colorspace == 'BGRx':
        bytes_per_px = 4
        assert(size == (width * height * bytes_per_px))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = bytes_per_px * (w + width * h)
                graphics += convert_to_bytes(data[pos])
                graphics += convert_to_bytes(data[pos + 1])
                graphics += convert_to_bytes(data[pos + 2])
            for x in range(0, (width * 3) % 4):
                graphics += pack('<B', 0)
    elif colorspace == 'GRAY8':
        bytes_per_px = 1
        assert(size == (width * height * bytes_per_px))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = bytes_per_px * (w + width * h)
                graphics += convert_to_bytes(data[pos])
            for x in range(0, (width * 3) % 4):
                graphics += pack('<B', 0)
    else:
        print('Unrecognized colorspace %', colorspace)
        sys.exit(1)

    # BMP file header
    if colorspace == 'GRAY8':
        header = pack('<HLHHL', 19778, (26 + width * height), 0, 0, 26)
        header += pack('<LHHHH', 12, width, height, 1, 8)
    else:
        header = pack('<HLHHL', 19778, (26 + width * height * 3), 0, 0, 26)
        header += pack('<LHHHH', 12, width, height, 1, 24)

    with open(filename, 'wb') as file:
        file.write(header)
        file.write(graphics)


##
# @brief Write the generated data
#
def write(filename, data):
    with open(filename, 'wb') as file:
        file.write(data)


##
# @brief Generate Golden Test Case, a single videosrctest frame of 280x40xRGB
# @return (string, string_size, expected_size)
#
# If string_size < expected_size, you do not need to check the results offset >= string_size.
# string: binary string (b'\xff\x00....')
def gen_RGB():
    string = b''
    string_size = 0
    expected_size = 280 * 40 * 3
    for i in range(0, 26):
        # White
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 255, 255, 255)
        # Yellow
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 255, 255, 0)
        # Light Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 255, 255)
        # Green
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 255, 0)
        # Purple
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 255, 0, 255)
        # Red
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 255, 0, 0)
        # Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 0, 255)
    for i in range(26, 30):
        # Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 0, 0)
        # Purple
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 255, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 0, 0)
        # Light Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 255, 255)
        # Black
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 0, 0, 0)
        # White
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBB', 255, 255, 255)
    for i in range(0, 46):
        # Dark Blue
        string_size = string_size + 46
        string += pack('BBB', 0, 0, 128)
    for i in range(46, 93):
        # White
        string_size = string_size + 47
        string += pack('BBB', 255, 255, 255)
    for i in range(93, 140):
        # Gray Blue
        string_size = string_size + 47
        string += pack('BBB', 0, 128, 255)
    for i in range(140, 186):
        # Black
        string_size = string_size + 46
        string += pack('BBB', 0, 0, 0)
    for i in range(186, 210):
        # Dark Gray
        string_size = string_size + 24
        string += pack('BBB', 19, 19, 19)
    # We do not check the reset pixels: they are randomly generated.
    string_size = string_size * 3
    return string, string_size, expected_size


##
# @brief Generate Golden Test Case, a single videosrctest frame of 280x40xBGRx
# @return (string, string_size, expected_size)
#
def gen_BGRx():
    string = b''
    string_size = 0
    expected_size = 280 * 40 * 4
    for i in range(0, 26):
        # White
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 255, 255, 255)
        # Yellow
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 0, 255, 255, 255)
        # Light Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 255, 0, 255)
        # Green
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 0, 255, 0, 255)
        # Purple
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 0, 255, 255)
        # Red
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 0, 0, 255, 255)
        # Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 0, 0, 255)
    for i in range(26, 30):
        # Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 0, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 0, 0, 0, 255)
        # Purple
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 0, 255, 255)
        # Black
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 0, 0, 0, 255)
        # Light Blue
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 255, 0, 255)
        # Black
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 0, 0, 0, 255)
        # White
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('BBBB', 255, 255, 255, 255)
    for i in range(0, 46):
        # Dark Blue
        string_size = string_size + 46
        string += pack('BBBB', 128, 0, 0, 255)
    for i in range(46, 93):
        # White
        string_size = string_size + 47
        string += pack('BBBB', 255, 255, 255, 255)
    for i in range(93, 140):
        # Gray Blue
        string_size = string_size + 47
        string += pack('BBBB', 255, 128, 0, 255)
    for i in range(140, 186):
        # Black
        string_size = string_size + 46
        string += pack('BBBB', 0, 0, 0, 255)
    for i in range(186, 210):
        # Dark Gray
        string_size = string_size + 24
        string += pack('BBBB', 19, 19, 19, 255)
    # We do not check the reset pixels: they are randomly generated.
    string_size = string_size * 4
    return string, string_size, expected_size


##
# @brief Generate Golden Test Case, a single videosrctest frame of 280x40xGRAY8
# @return (string, string_size, expected_size)
#
# If string_size < expected_size, you do not need to check the results offset >= string_size.
# string: binary string (b'\xff\x00....')
def gen_GRAY8():
    string = b''
    string_size = 0
    expected_size = 280 * 40
    for i in range(0, 26):
        # 0xEB
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 235)
        # 0xD2
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 210)
        # 0xAA
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 170)
        # 0x91
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 145)
        # 0x6A
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 106)
        # 0x51
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 81)
        # 0x29
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 41)
    for i in range(26, 30):
        # 0x29
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 41)
        # 0x10
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 16)
        # 0x6A
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 106)
        # 0x10
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 16)
        # 0xAA
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 170)
        # 0x10
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 16)
        # 0xEB
        string_size = string_size + 40
        for j in range(0, 40):
            string += pack('B', 235)
    for i in range(0, 46):
        # 0x10
        string_size = string_size + 46
        string += pack('B', 16)
    for i in range(46, 93):
        # 0xEB
        string_size = string_size + 47
        string += pack('B', 235)
    for i in range(93, 140):
        # 0x10
        string_size = string_size + 47
        string += pack('B', 16)
    for i in range(140, 163):
        # 0x00
        string_size = string_size + 23
        string += pack('B', 0)
    for i in range(163, 186):
        # 0x10
        string_size = string_size + 23
        string += pack('B', 16)
    for i in range(186, 210):
        # 0x20
        string_size = string_size + 24
        string += pack('B', 32)
    # We do not check the reset pixels: they are randomly generated.
    return string, string_size, expected_size

##
# @brief Generate Golden Test Case, a randomly generated BMP image
# @return (string, string_size, expected_size)
#
def gen_BMP_random(color_type, width, height, filename_prefix):
    string = b''
    string_size = 0
    size_per_pixel = 3
    if color_type == 'BGRx':
        size_per_pixel = 4
    elif color_type == "GRAY8":
        size_per_pixel = 1
    expected_size = width * height * size_per_pixel
    # The result has no stride for other/tensor types.

    if color_type == 'BGRx':
        for y in range(0, height):
            for x in range(0, width):
                pval = (random.randrange(256), random.randrange(256), random.randrange(256))
                pixel = pack('BBBB', pval[2], pval[1], pval[0], 255)
                string += pixel
                string_size += 4
    elif color_type == 'GRAY8':
        for y in range(0, height):
            for x in range(0, width):
                pval = random.randrange(256)
                pixel = pack('B', pval)
                string += pixel
                string_size += size_per_pixel
    else:
        # Assume RGB
        for y in range(0, height):
            for x in range(0, width):
                pval = (random.randrange(256), random.randrange(256), random.randrange(256))
                pixel = pack('BBB', pval[0], pval[1], pval[2])
                string += pixel
                string_size += 3

    saveBMP(filename_prefix + '_' + color_type + '_' + str(width) + 'x' + str(height) + '.bmp',
            string, color_type, width, height)
    return string, string_size, expected_size


##
# @brief Generate a fixed BMP sequence for stream test
# @return 0 if success. non-zero if failed.
#
# This gives "16x16", black, white, green, red, blue, wb-checker, rb-checker, gr-checker,
# red-cross-on-white, blue-cross-on-black (4x4 x 16, left-top/right-bottom white/red/green).
# "10 files" with 0 ~ 9 postfix in the filename
def gen_BMP_stream(filename_prefix, golden_filename, num_sink):
    string = [b'' for _ in range(10)]
    size_x = 16
    size_y = 16

    for y in range(0, size_y):
        for x in range(0, size_x):
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

    with open(golden_filename, 'wb') as file:
        for i in range(0, 10):
            saveBMP(filename_prefix + '_' + str(i) + '.bmp', string[i], 'RGB', 16, 16)
            for j in range(0, num_sink):
                file.write(string[i])
    return string
