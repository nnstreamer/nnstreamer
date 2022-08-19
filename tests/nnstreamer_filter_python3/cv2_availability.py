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
  except ImportError:
    return 1

  return 0

sys.exit(main())
