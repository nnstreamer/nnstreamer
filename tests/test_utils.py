#!/usr/bin/env python
# SPDX-License-Identifier: LGPL-2.1-only
#
# Utility functions to run the unittests
# Copyright (c) 2021 Samsung Electronics Co., Ltd.
#
# @file   test_utils.py
# @brief  Utility functions to run the unittests
# @author Jaeyun Jung <jy1210.jung@samsung.com>
# @date   20 Aug 2021
# @bug    No known bugs

import sys
import os


##
# @brief Read file and return its content
def read_file(filename):
    with open(filename, 'rb') as f:
        b = f.read()
    return b


##
# @brief Read labels from file
def read_label(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines
