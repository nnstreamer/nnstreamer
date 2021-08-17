##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2019 Samsung Electronics
#
# @file    passthrough.py
# @brief   Python custom filter example: passthrough
# @author  Dongju Chae <dongju.chae@samsung.com>

import numpy as np
import nnstreamer_python as nns

D1 = 3
D2 = 280
D3 = 40
D4 = 1


##
# @brief  User-defined custom filter; DO NOT CHANGE CLASS NAME
class CustomFilter(object):
    ##
    # @brief  The constructor for custom filter: passthrough
    def __init__(self, *args):
        self.input_dims = [nns.TensorShape([D1, D2, D3, D4], np.uint8)]
        self.output_dims = [nns.TensorShape([D1, D2, D3, D4], np.uint8)]

    ##
    # @brief  python callback: getInputDim
    # @param  None
    # @return user-assigned input dimensions
    def getInputDim(self):
        return self.input_dims

    ##
    # @brief  Python callback: getOutputDim
    # @param  None
    # @return user-assigned output dimensions
    def getOutputDim(self):
        return self.output_dims

    ##
    # @brief  Python callback: invoke
    # @param  Input tensors: list of input numpy array
    # @return output tensors: list of output numpy array
    def invoke(self, input_array):
        # passthrough, just return.
        return input_array
