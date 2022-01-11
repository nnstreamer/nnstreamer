##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2021 Samsung Electronics
#
# @file    custom_converter_json.py
# @brief   Python custom converter
# @author  Gichan Jang <gichan2.jang@samsung.com>

import numpy as np
import nnstreamer_python as nns
import json


##
# @brief  User-defined custom converter
class CustomConverter(object):
    ##
    # @brief  Python callback: convert
    #
    # input:
    # {
    #     "json_string": "string_example",
    #     "json_number": 100,
    #     "json_array": [1, 2, 3, 4, 5],
    #     "json_object": {"name":"John", "age":30},
    #     "json_bool": true
    # }
    #
    # output:
    # tensor 0 : "string_example"
    # tensor 1 : {"name":"John", "age":30}
    def convert(self, input_array):
        json_data = json.loads(input_array[0].tobytes())
        json_string = (json_data["json_string"] + '\0').encode()
        json_object = json.dumps(json_data["json_object"]).encode()

        output_array1 = np.frombuffer(json_string, dtype=np.uint8)
        output_array2 = np.frombuffer(json_object, dtype=np.uint8)
        raw_data = [output_array1, output_array2]

        rate_n = 10
        rate_d = 1
        dim1 = [len(raw_data[0]), 1, 1, 1]
        dim2 = [len(raw_data[1]), 1, 1, 1]
        ttype = np.uint8
        tensors_info = [nns.TensorShape(dim1, ttype), nns.TensorShape(dim2, ttype)]

        return tensors_info, raw_data, rate_n, rate_d
