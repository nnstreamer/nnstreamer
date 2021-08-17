##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2021 Samsung Electronics
#
# @file    custom_converter.py
# @brief   Python custom converter
# @author  Gichan Jang <gichan2.jang@samsung.com>
#
# @note The Flexbuffers Python API is supported if the flatbuffers version is greater than 1.12.
#       See https://github.com/google/flatbuffers/issues/5306. It can be downloaded and used from
#       https://github.com/google/flatbuffers/blob/master/python/flatbuffers/flexbuffers.py.

import numpy as np
import nnstreamer_python as nns
from flatbuffers import flexbuffers


##
# @brief  Change from numpy type to tensor type
def convert_to_numpy_type(dtype):
    if dtype == 0:
        return np.int32
    elif dtype == 1:
        return np.uint32
    elif dtype == 2:
        return np.int16
    elif dtype == 3:
        return np.uint16
    elif dtype == 4:
        return np.int8
    elif dtype == 5:
        return np.uint8
    elif dtype == 6:
        return np.float64
    elif dtype == 7:
        return np.float32
    elif dtype == 8:
        return np.int64
    elif dtype == 9:
        return np.uint64
    else:
        print("Not supported numpy type")
        return -1


##
# @brief  User-defined custom converter
class CustomConverter(object):
    ##
    # @brief  Python callback: convert
    def convert(self, input_array):
        data = input_array[0].tobytes()
        root = flexbuffers.GetRoot(data)
        tensors = root.AsMap

        num_tensors = tensors['num_tensors'].AsInt
        rate_n = tensors['rate_n'].AsInt
        rate_d = tensors['rate_d'].AsInt
        raw_data = []
        tensors_info = []

        for i in range(num_tensors):
            tensor_key = "tensor_{idx}".format(idx=i)
            tensor = tensors[tensor_key].AsVector
            ttype = convert_to_numpy_type(tensor[1].AsInt)
            tdim = tensor[2].AsTypedVector
            dim = []
            for j in range(4):
                dim.append(tdim[j].AsInt)
            tensors_info.append(nns.TensorShape(dim, ttype))
            raw_data.append(np.frombuffer(tensor[3].AsBlob, dtype=np.uint8))

        return tensors_info, raw_data, rate_n, rate_d
