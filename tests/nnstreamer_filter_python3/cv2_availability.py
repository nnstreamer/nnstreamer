#!/bin/env python3
##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2022 Samsung Electronics
#
# @file    cv2_availability.py
# @brief   Check if cv2 is available for python
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>

import sys

def main():
  try:
    import cv2
    print("USE_CV2:True", __file__)
    return 0
  except ImportError:
    print("OpenCV is not available, but continuing...")
    print("USE_CV2:False", __file__)
    return 0

sys.exit(main())
