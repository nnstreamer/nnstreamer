##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2021 Samsung Electronics
#
# @file    custom_converter_test.py
# @brief   Python custom converter
# @author  Gichan Jang <gichan2.jang@samsung.com>

import numpy as np
import nnstreamer_python as nns


##
# @brief  User-defined custom converter
class CustomConverter(object):
    ##
    # @brief  Python callback: convert
    def convert(self, input_array):
        rate_n = 10
        rate_d = 1
        dim = [len(input_array[0]) / 4, 1, 1, 1]
        ttype = np.int32
        tensors_info = [nns.TensorShape(dim, ttype)]

        return tensors_info, input_array, rate_n, rate_d
