##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2021 Samsung Electronics
#
# @file    custom_decoder.py
# @brief   Python custom decoder
# @author  Gichan Jang <gichan2.jang@samsung.com>
#
# @note The Flexbuffers Python API is supported if the flatbuffers version is greater than 1.12.
#       See: https://github.com/google/flatbuffers/issues/5306. It can be downloaded and used from
#       https://github.com/google/flatbuffers/blob/master/python/flatbuffers/flexbuffers.py.

import numpy as np
import nnstreamer_python as nns
from flatbuffers import flexbuffers


##
# @brief  Change from numpy type to tensor type
def convert_to_tensor_type(dtype):
    if dtype == np.int32:
        return 0
    elif dtype == np.uint32:
        return 1
    elif dtype == np.int16:
        return 2
    elif dtype == np.uint16:
        return 3
    elif dtype == np.int8:
        return 4
    elif dtype == np.uint8:
        return 5
    elif dtype == np.float64:
        return 6
    elif dtype == np.float32:
        return 7
    elif dtype == np.int64:
        return 8
    elif dtype == np.uint64:
        return 9
    else:
        print("Not supported tensor type")
        return -1


##
# @brief  User-defined custom decoder
class CustomDecoder(object):
    ##
    # @brief  Python callback: getOutCaps
    def getOutCaps(self):
        return bytes('other/flexbuf', 'UTF-8')

    ##
    # @brief  Python callback: decode
    def decode(self, raw_data, in_info, rate_n, rate_d):
        fbb = flexbuffers.Builder()
        num_tensors = len(in_info)

        with fbb.Map():
            fbb.UInt("num_tensors", num_tensors)
            fbb.Int("rate_n", rate_n)
            fbb.Int("rate_d", rate_d)
            for i in range(num_tensors):
                tensor_key = "tensor_{idx}".format(idx=i)
                dtype = in_info[i].getType().type
                ttype = convert_to_tensor_type(dtype)
                dims = in_info[i].getDims()
                with fbb.Vector(tensor_key):
                    fbb.String("")
                    fbb.Int(ttype)
                    with fbb.TypedVector():
                        for j in range(4):
                            fbb.Int(dims[j])
                    fbb.Blob(raw_data[i])

        data = bytes(fbb.Finish())
        return data
