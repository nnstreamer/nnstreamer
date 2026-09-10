##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    converter_output_cases.py
# @brief   Python custom converter returning well-formed and ill-formed outputs
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# @note    The first byte of the input selects what convert returns.

import numpy as np
import nnstreamer_python as nns


##
# @brief  User-defined custom converter
class CustomConverter:
    ##
    # @brief  Python callback: convert
    # @param  input_array  Input data: list of uint8 numpy array
    # @return tensors info, tensors, framerate of the requested case
    def convert(self, input_array):
        mode = int(input_array[0][0])
        int32x4 = [nns.TensorShape([4], np.int32)]
        seq = np.arange(8, dtype=np.int32)
        cases = {
            1: (int32x4, [seq[:4].copy()]),
            2: (int32x4, [seq[3::-1]]),
            4: (int32x4, [seq[:4].tobytes()]),
            5: (int32x4, [seq[:2].copy()]),
            6: (int32x4 * 2, [seq[:4].copy()]),
            7: (int32x4, (seq[:4].copy(),)),
        }

        if mode == 3:
            raise ValueError('mode 3 raises')
        if mode in cases:
            tensors_info, tensors = cases[mode]
        else:
            tensors_info = [nns.TensorShape([len(input_array[0])], np.uint8)]
            tensors = [input_array[0]]
        return tensors_info, tensors, 10, 1
