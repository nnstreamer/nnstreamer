##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    gen_add_one_224.py
# @brief   Generate add_one_224.onnx, a 3:224:224:1 float32 "+1.0" model
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# The model is a single 1x1 convolution whose kernel is the 3x3 identity and
# whose bias is 1.0 on every channel, so it computes exactly out = in + 1.0
# on a 1x3x224x224 (NCHW) float32 tensor.
#
# It exists so that sub-plugins backed by a compiling runtime (TensorRT, which
# builds an .engine from the .onnx) have a positive test fixture that is both
# tiny and byte-exact comparable:
#
#   - Only Conv is used. Quantized ONNX models in this directory store their
#     weights as uint8 behind DequantizeLinear, which the TensorRT ONNX parser
#     rejects (it accepts int8/fp8/int4 quantized tensors only).
#   - The batch dimension is fixed at 1; TensorRT sub-plugins in this tree
#     reject dynamic batch sizes.
#   - Multiplying by 1.0 and adding 1.0 is exact in IEEE-754 for the integral
#     pixel values (0..255) the pipelines feed in, and stays exact under the
#     reduced-precision TF32 accumulation TensorRT enables by default, whose
#     11-bit significand represents every integer below 2048. A pipeline can
#     therefore compare the sub-plugin output byte for byte against a branch
#     that applies the same offset with tensor_transform.
#
# Usage: python3 gen_add_one_224.py   (in this directory)
# Requires: onnx, numpy

"""Generate add_one_224.onnx, an exact elementwise +1.0 model for filter tests."""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

DST = 'add_one_224.onnx'
CHANNELS = 3
SIDE = 224


def build_model():
    """Return the ONNX model computing out = in + 1.0 via an identity 1x1 conv."""
    shape = [1, CHANNELS, SIDE, SIDE]
    weight = np.eye(CHANNELS, dtype=np.float32).reshape(CHANNELS, CHANNELS, 1, 1)
    bias = np.ones(CHANNELS, dtype=np.float32)

    node = helper.make_node(
        'Conv', ['input', 'weight', 'bias'], ['output'], name='add_one',
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1)
    graph = helper.make_graph(
        [node], 'add_one_224',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)],
        initializer=[numpy_helper.from_array(weight, 'weight'),
                     numpy_helper.from_array(bias, 'bias')])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=7)


def main():
    """Build the model, check it, and save it."""
    model = build_model()
    onnx.checker.check_model(model)
    onnx.save(model, DST)
    print('saved', DST)


if __name__ == '__main__':
    main()
