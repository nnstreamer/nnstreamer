#!/usr/bin/env python

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2018 Samsung Electronics
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
# @param[in] data list of numbers that would be saved as BMP format
# @param[in] colorspace "RGB" or "BGRx"
# @param[in] width Width of the picture
# @param[in] height Height of the picture
def saveBMP(filename, data, colorspace, width, height):
    size = len(data)
    pixel_list = []
    # Default value of bytes per pixel value for RGB
    bytes_per_px = 3

    if colorspace == 'RGB':
        assert(size == (width * height * bytes_per_px))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = 3 * (w + width * h)
                pixel_list.append(data[pos + 2])
                pixel_list.append(data[pos + 1])
                pixel_list.append(data[pos + 0])
            for x in range(0, (width * 3) % 4):
                pixel_list.append(0)
    elif colorspace == 'BGRx':
        bytes_per_px = 4
        assert(size == (width * height * bytes_per_px))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = bytes_per_px * (w + width * h)
                pixel_list.append(data[pos + 0])
                pixel_list.append(data[pos + 1])
                pixel_list.append(data[pos + 2])
            for x in range(0, (width * 3) % 4):
                pixel_list.append(0)
    elif colorspace == 'GRAY8':
        bytes_per_px = 1
        assert(size == (width * height * bytes_per_px))
        # BMP is stored bottom to top. Reverse the order
        for h in range(height-1, -1, -1):
            for w in range(0, width):
                pos = bytes_per_px * (w + width * h)
                pixel_list.append(data[pos])
            for x in range(0, (width * 3) % 4):
                pixel_list.append(0)
    else:
        print('Unrecognized colorspace %', colorspace)
        sys.exit(1)

    graphics = pack('%dB' % (len(pixel_list)), *pixel_list)
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
    expected_size = 280 * 40 * 3
    pixels = []
    for i in range(0, 26):
        # White
        for j in range(0, 40):
            pixels += [255, 255, 255]
        # Yellow
        for j in range(0, 40):
            pixels += [255, 255, 0]
        # Light Blue
        for j in range(0, 40):
            pixels += [0, 255, 255]
        # Green
        for j in range(0, 40):
            pixels += [0, 255, 0]
        # Purple
        for j in range(0, 40):
            pixels += [255, 0, 255]
        # Red
        for j in range(0, 40):
            pixels += [255, 0, 0]
        # Blue
        for j in range(0, 40):
            pixels += [0, 0, 255]
    for i in range(26, 30):
        # Blue
        for j in range(0, 40):
            pixels += [0, 0, 255]
        # Black
        for j in range(0, 40):
            pixels += [0, 0, 0]
        # Purple
        for j in range(0, 40):
            pixels += [255, 0, 255]
        # Black
        for j in range(0, 40):
            pixels += [0, 0, 0]
        # Light Blue
        for j in range(0, 40):
            pixels += [0, 255, 255]
        # Black
        for j in range(0, 40):
            pixels += [0, 0, 0]
        # White
        for j in range(0, 40):
            pixels += [255, 255, 255]
    for i in range(0, 46):
        # Dark Blue
        pixels += [0, 0, 128]
    for i in range(46, 93):
        # White
        pixels += [255, 255, 255]
    for i in range(93, 140):
        # Gray Blue
        pixels += [0, 128, 255]
    for i in range(140, 186):
        # Black
        pixels += [0, 0, 0]
    for i in range(186, 210):
        # Dark Gray
        pixels += [19, 19, 19]
    # We do not check the reset pixels: they are randomly generated.
    string_size = len(pixels)
    string = pack('%dB' % (len(pixels)), *pixels)
    return string, string_size, expected_size


##
# @brief Generate Golden Test Case, a single videosrctest frame of 280x40xBGRx
# @return (string, string_size, expected_size)
#
def gen_BGRx():
    pixels = []
    expected_size = 280 * 40 * 4
    for i in range(0, 26):
        # White
        for j in range(0, 40):
            pixels += [255, 255, 255, 255]
        # Yellow
        for j in range(0, 40):
            pixels += [0, 255, 255, 255]
        # Light Blue
        for j in range(0, 40):
            pixels += [255, 255, 0, 255]
        # Green
        for j in range(0, 40):
            pixels += [0, 255, 0, 255]
        # Purple
        for j in range(0, 40):
            pixels += [255, 0, 255, 255]
        # Red
        for j in range(0, 40):
            pixels += [0, 0, 255, 255]
        # Blue
        for j in range(0, 40):
            pixels += [255, 0, 0, 255]
    for i in range(26, 30):
        # Blue
        for j in range(0, 40):
            pixels += [255, 0, 0, 255]
        # Black
        for j in range(0, 40):
            pixels += [0, 0, 0, 255]
        # Purple
        for j in range(0, 40):
            pixels += [255, 0, 255, 255]
        # Black
        for j in range(0, 40):
            pixels += [0, 0, 0, 255]
        # Light Blue
        for j in range(0, 40):
            pixels += [255, 255, 0, 255]
        # Black
        for j in range(0, 40):
            pixels += [0, 0, 0, 255]
        # White
        for j in range(0, 40):
            pixels += [255, 255, 255, 255]
    for i in range(0, 46):
        # Dark Blue
        pixels += [128, 0, 0, 255]
    for i in range(46, 93):
        # White
        pixels += [255, 255, 255, 255]
    for i in range(93, 140):
        # Gray Blue
        pixels += [255, 128, 0, 255]
    for i in range(140, 186):
        # Black
        pixels += [0, 0, 0, 255]
    for i in range(186, 210):
        # Dark Gray
        pixels += [19, 19, 19, 255]
    # We do not check the reset pixels: they are randomly generated.
    string_size = len(pixels)
    string = pack('%dB' % (len(pixels)), *pixels)
    return string, string_size, expected_size


##
# @brief Generate Golden Test Case, a single videosrctest frame of 280x40xGRAY8
# @return (string, string_size, expected_size)
#
# If string_size < expected_size, you do not need to check the results offset >= string_size.
# string: binary string (b'\xff\x00....')
def gen_GRAY8():
    pixels = []
    expected_size = 280 * 40
    for i in range(0, 26):
        # 0xEB
        for j in range(0, 40):
            pixels += [235]
        # 0xD2
        for j in range(0, 40):
            pixels += [210]
        # 0xAA
        for j in range(0, 40):
            pixels += [170]
        # 0x91
        for j in range(0, 40):
            pixels += [145]
        # 0x6A
        for j in range(0, 40):
            pixels += [106]
        # 0x51
        for j in range(0, 40):
            pixels += [81]
        # 0x29
        for j in range(0, 40):
            pixels += [41]
    for i in range(26, 30):
        # 0x29
        for j in range(0, 40):
            pixels += [41]
        # 0x10
        for j in range(0, 40):
            pixels += [16]
        # 0x6A
        for j in range(0, 40):
            pixels += [106]
        # 0x10
        for j in range(0, 40):
            pixels += [16]
        # 0xAA
        for j in range(0, 40):
            pixels += [170]
        # 0x10
        for j in range(0, 40):
            pixels += [16]
        # 0xEB
        for j in range(0, 40):
            pixels += [235]
    for i in range(0, 46):
        # 0x10
        pixels += [16]
    for i in range(46, 93):
        # 0xEB
        pixels += [235]
    for i in range(93, 140):
        # 0x10
        pixels += [16]
    for i in range(140, 163):
        # 0x00
        pixels += [0]
    for i in range(163, 186):
        # 0x10
        pixels += [16]
    for i in range(186, 210):
        # 0x20
        pixels += [32]
    # We do not check the reset pixels: they are randomly generated.
    string_size = len(pixels)
    string = pack('%dB' % (len(pixels)), *pixels)
    return string, string_size, expected_size

##
# @brief Generate Golden Test Case, a randomly generated BMP image
# @return (string, string_size, expected_size)
#
def gen_BMP_random(color_type, width, height, filename_prefix):
    pixel_list = []
    size_per_pixel = 3
    if color_type == 'BGRx':
        size_per_pixel = 4
    elif color_type == "GRAY8":
        size_per_pixel = 1
    expected_size = width * height * size_per_pixel
    # The result has no stride for other/tensor types.

    if color_type == 'BGRx':
        for _ in range(height * width):
            pixel_list += [random.randrange(256), random.randrange(256), random.randrange(256), 255]
    elif color_type == 'GRAY8':
        pixel_list = [random.randrange(256) for _ in range(height * width * 1)]
    else:
        # Assume RGB
        pixel_list = [random.randrange(256) for _ in range(height * width * 3)]

    saveBMP(filename_prefix + '_' + color_type + '_' + str(width) + 'x' + str(height) + '.bmp',
            pixel_list, color_type, width, height)

    string_size = len(pixel_list)
    string = pack('%dB' % (string_size), *pixel_list)

    return string, string_size, expected_size


##
# @brief Generate a fixed BMP sequence for stream test
# @return 0 if success. non-zero if failed.
#
# This gives "16x16", black, white, green, red, blue, wb-checker, rb-checker, gr-checker,
# red-cross-on-white, blue-cross-on-black (4x4 x 16, left-top/right-bottom white/red/green).
# "10 files" with 0 ~ 9 postfix in the filename
def gen_BMP_stream(filename_prefix, golden_filename, num_sink):
    pixels = [[] for _ in range(10)]
    size_x = 16
    size_y = 16

    for y in range(0, size_y):
        for x in range(0, size_x):
            # black. Frame 0
            pixels[0] += [0, 0, 0]
                # white. Frame 1
            pixels[1] += [255, 255, 255]
                # green, Frame 2
            pixels[2] += [0, 255, 0]
                # red, Frame 3
            pixels[3] += [255, 0, 0]
                # blue, Frame 4
            pixels[4] += [0, 0, 255]
            # white-black checker, Frame 5
            if (((x / 4) % 2) + ((y / 4) % 2)) == 1:
                pixels[5] += [0, 0, 0]
            else:
                pixels[5] += [255, 255, 255]
            # red-blue checker, Frame 6
            if (((x / 4) % 2) + ((y / 4) % 2)) == 1:
                pixels[6] += [0, 0, 255]
            else:
                pixels[6] += [255, 0, 0]
            # green-red checker, Frame 7
            if (((x / 4) % 2) + ((y / 4) % 2)) == 1:
                pixels[7] += [255, 0, 0]
            else:
                pixels[7] += [0, 255, 0]
            # red-cross-on-white, Frame 8
            if x == y:
                pixels[8] += [255, 0, 0]
            else:
                pixels[8] += [255, 255, 255]
            # blue-cross-on-black, Frame 9
            if x == y:
                pixels[9] += [0, 0, 255]
            else:
                pixels[9] += [0, 0, 0]

    with open(golden_filename, 'wb') as file:
        for i in range(0, 10):
            saveBMP(filename_prefix + '_' + str(i) + '.bmp', pixels[i], 'RGB', 16, 16)
            string = [ pack('%dB' % (len(v)), *v) for v in pixels ]
            for j in range(0, num_sink):
                file.write(string[i])
    return string
