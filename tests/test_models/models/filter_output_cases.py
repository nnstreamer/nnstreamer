##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    filter_output_cases.py
# @brief   Python custom filter returning well-formed and ill-formed outputs
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# @note    The first custom argument selects what invoke returns.
#          The filter takes a uint8 tensor of 8 and gives two uint8 tensors of 4.
#          A failing case fails only its first invoke and copies afterwards.

import numpy as np
import nnstreamer_python as nns


##
# @brief  User-defined custom filter; DO NOT CHANGE CLASS NAME
class CustomFilter:
    ##
    # @brief  The constructor for custom filter: output cases
    # @param  args  The custom arguments, the first one is the case to run
    def __init__(self, *args):
        self.mode = args[0] if args else 'copy'
        if self.mode == 'init_error':
            raise ValueError('init_error is requested')
        self.input_dims = [nns.TensorShape([8], np.uint8)]
        self.output_dims = [nns.TensorShape([4], np.uint8),
                            nns.TensorShape([4], np.uint8)]
        self.kept = np.arange(4, dtype=np.uint8)
        self.invoked = False

    ##
    # @brief  Python callback: getInputDim
    # @return user-assigned input dimensions
    def getInputDim(self):
        return self.input_dims

    ##
    # @brief  Python callback: getOutputDim
    # @return user-assigned output dimensions
    def getOutputDim(self):
        return self.output_dims

    ##
    # @brief  Python callback: invoke
    # @param  input_array  Input tensors: list of input numpy array
    # @return output tensors of the requested case
    def invoke(self, input_array):
        data = input_array[0]
        seq = np.arange(16, dtype=np.uint8)
        cases = {
            'alias': lambda: [data[:4], data[4:]],
            'strided': lambda: [seq[:8:2], seq[7::-2]],
            'same': lambda: [seq[:4]] * 2,
            'kept': lambda: [self.kept, data[4:].copy()],
            'not_array': lambda: [data[:4].tobytes(), data[4:].copy()],
            'not_list': lambda: (data[:4].copy(), data[4:].copy()),
            'count': lambda: [data[:4].copy()],
            'second_bad': lambda: [data[:4].copy(), data[4:].astype(np.int32)],
        }

        mode = self.mode
        if self.invoked and mode in ('raise', 'not_array', 'not_list', 'count',
                                     'second_bad'):
            mode = 'copy'
        self.invoked = True

        if mode == 'raise':
            raise ValueError('raise is requested')
        if mode in cases:
            return cases[mode]()
        return [data[:4].copy(), data[4:].copy()]
