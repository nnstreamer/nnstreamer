##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2019 Samsung Electronics
#
# @file    scaler.py
# @brief   Python custom filter example: scaler
# @author  Dongju Chae <dongju.chae@samsung.com>
# @note    This filter is an example of Python custom filter
#          and only supports rank 2,3 models.

import numpy as np
import nnstreamer_python as nns


##
# @brief  User-defined custom filter; DO NOT CHANGE CLASS NAME
class CustomFilter(object):
    new_x = 0
    new_y = 0
    input_dims = []
    output_dims = []

    ##
    # @brief  The constructor for custom filter: scaler
    # @param  Dimensions to scale
    def __init__(self, *args):
        if len(args) == 1:
            dims = args[0].split('x')
            if len(dims) != 2:
                raise RuntimeError("Dimension should have this format: AAAxBBB (e.g., 640x480)")

            (self.new_x, self.new_y) = [int(dim) for dim in dims]

    ##
    # @brief  Python callback: setInputDim
    # @param  Input dimensions: list of nns.TensorShape
    def setInputDim(self, input_dims):
        if len(input_dims) != 1:
            print("One input tensor is allowed")
            return None

        self.input_dims = input_dims
        self.output_dims = [nns.TensorShape(self.input_dims[0].getDims(), self.input_dims[0].getType())]

        dims = self.output_dims[0].getDims()
        if self.new_x > 0:
            dims[1] = self.new_x
        if self.new_y > 0:
            dims[2] = self.new_y

        return self.output_dims

    ##
    # @brief  Python callback: invoke
    # @param  Input tensors: list of input numpy array
    # @return Output tensors: list of output numpy array
    def invoke(self, input_array):
        # reshape to n-D array (in reverse order)
        in_dims = self.input_dims[0].getDims()
        out_dims = self.output_dims[0].getDims()

        reversed_in_dims = np.flip(in_dims,axis=0)
        reversed_out_dims = np.flip(out_dims,axis=0)

        input_tensor = np.reshape(input_array[0], reversed_in_dims[~np.in1d(reversed_in_dims, np.array([0, 1]))])
        output_tensor = np.empty(reversed_out_dims[~np.in1d(reversed_out_dims, np.array([0, 1]))], dtype=self.output_dims[0].getType())

        if len(output_tensor.shape) == 3:
            for y in range(out_dims[2]):
                for x in range(out_dims[1]):
                    for c in range(out_dims[0]):
                        ix = int(x * in_dims[1] / out_dims[1])
                        iy = int(y * in_dims[2] / out_dims[2])
                        output_tensor[y][x][c] = input_tensor[iy][ix][c]
        elif len(output_tensor.shape) == 2:
            for x in range(out_dims[1]):
                for c in range(out_dims[0]):
                    ix = int(x * in_dims[1] / out_dims[1])
                    iy = int(in_dims[2] / out_dims[2])
                    output_tensor[x][c] = input_tensor[ix][c]

        # to 1-D array
        return [np.ravel(output_tensor)]
