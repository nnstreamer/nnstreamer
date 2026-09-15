##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    custom_decoder_raw.py
# @brief   Python custom decoder concatenating the raw tensors, with no module beyond numpy
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# @note    NNS_TEST_PY_DECODER_MODE, read when an instance is created, selects what decode does:
#          concat (default) checks raw_data against in_info and returns the tensors as bytes,
#          raise raises an exception, fixed returns FIXED.

import os

import numpy as np

FIXED = bytes([1, 2, 3, 4])


##
# @brief  User-defined custom decoder
class CustomDecoder:
    ##
    # @brief  The constructor, which reads the mode
    def __init__(self):
        self.mode = os.environ.get('NNS_TEST_PY_DECODER_MODE', 'concat')

    ##
    # @brief  Python callback: getOutCaps
    # @return the output caps
    def getOutCaps(self):
        return b'application/octet-stream'

    ##
    # @brief  Python callback: decode
    # @param  raw_data  List of the input tensors as 1-D uint8 numpy arrays
    # @param  in_info  List of nns.TensorShape of the input tensors
    # @param  rate_n  Numerator of the framerate
    # @param  rate_d  Denominator of the framerate
    # @return the decoded bytes
    def decode(self, raw_data, in_info, rate_n, rate_d):
        if self.mode == 'raise':
            raise RuntimeError('raise is requested')
        if self.mode == 'fixed':
            return FIXED
        if len(raw_data) != len(in_info):
            raise ValueError('raw_data and in_info disagree')
        for data, info in zip(raw_data, in_info):
            dims = [d for d in info.getDims() if d > 0]
            if data.size != int(np.prod(dims)):
                raise ValueError('a tensor does not match its dimensions')
        return b''.join(data.tobytes() for data in raw_data)
