##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    gen_mobilenet_v2_float.py
# @brief   Generate mobilenet_v2_float.onnx from mobilenet_v2_quant.onnx
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Inference results of the quantized (QOperator) model depend on the CPU the
# test runs on: onnxruntime dispatches to different int8 kernels per CPU
# feature set, and the AVX2-without-VNNI path may saturate its intermediate
# 16-bit accumulations (vpmaddubsw), which can change the argmax of the output
# (e.g., GitHub-hosted CI runners are a mix of Intel VNNI-capable Xeons and
# AMD EPYC without VNNI, so classification tests on the quantized model fail
# sporadically). This script rewrites the model to compute in float32, which
# is numerically stable across kernel paths, while keeping the weights stored
# as uint8 (DequantizeLinear) so the file stays small.
#
# Each QLinear* op is replaced with its float counterpart, followed by a Clip
# to the op's u8 output range, which reproduces the fused activation (ReLU6)
# encoded in the quantization parameters.
#
# Usage: python3 gen_mobilenet_v2_float.py   (in this directory)
# Requires: onnx, numpy

"""Generate mobilenet_v2_float.onnx (float compute) from mobilenet_v2_quant.onnx."""

import numpy as np
import onnx
from onnx import helper, numpy_helper

SRC = 'mobilenet_v2_quant.onnx'
DST = 'mobilenet_v2_float.onnx'


class QuantToFloatConverter:
    """Rewrite a QOperator-format quantized graph into a float-compute graph."""

    def __init__(self, graph):
        """Prepare the conversion state for the given source graph."""
        self.graph = graph
        self.inits = {init.name: init for init in graph.initializer}
        self.new_nodes = []
        self.new_inits = {}
        self.fmap = {}  # quantized tensor name -> float tensor name
        self.uid = 0

    def next_uid(self):
        """Return a unique number for generated node/initializer names."""
        self.uid += 1
        return self.uid

    def keep_init(self, name):
        """Carry an original initializer over to the converted graph."""
        if name not in self.new_inits:
            self.new_inits[name] = self.inits[name]
        return name

    def fname(self, name):
        """Return the float tensor name for a (possibly quantized) tensor."""
        return self.fmap.get(name, name)

    def to_arr(self, init_name):
        """Return an initializer's value as a numpy array."""
        return numpy_helper.to_array(self.inits[init_name])

    def add_float_init(self, name, arr):
        """Add a float32 initializer with the given value."""
        self.new_inits[name] = numpy_helper.from_array(arr.astype(np.float32), name)
        return name

    def dequantize(self, w_name, s_name, zp_name, axis=None):
        """Emit DequantizeLinear(w) -> float, keeping u8 weight storage."""
        out = f'{w_name}_f32'
        kwargs = {} if axis is None else {'axis': axis}
        self.new_nodes.append(helper.make_node(
            'DequantizeLinear',
            [self.keep_init(w_name), self.keep_init(s_name), self.keep_init(zp_name)],
            [out], name=f'dq_{self.next_uid()}', **kwargs))
        return out

    def emit_clipped(self, node, y_scale, y_zp):
        """Emit the node with a Clip to its u8 output range appended.

        The quantized op clamps its requantized output to [0, 255] in u8
        space, which encodes the fused activation (e.g. ReLU6).
        """
        uid = self.next_uid()
        scale = float(self.to_arr(y_scale))
        zero = float(self.to_arr(y_zp))
        low = self.add_float_init(f'clip_min_{uid}', np.array((0.0 - zero) * scale))
        high = self.add_float_init(f'clip_max_{uid}', np.array((255.0 - zero) * scale))
        out = node.output[0]
        pre = out + '_preclip'
        node.output[0] = pre
        self.new_nodes.append(node)
        self.new_nodes.append(helper.make_node('Clip', [pre, low, high], [out], name=f'clip_{uid}'))

    def convert_conv(self, node):
        """Convert QLinearConv to Conv with dequantized weights and folded bias."""
        x_name, x_scale, _, w_name, w_scale, w_zp, y_scale, y_zp = node.input[:8]
        w_scale_arr = self.to_arr(w_scale).astype(np.float32)
        axis = 0 if w_scale_arr.ndim > 0 and w_scale_arr.size > 1 else None
        inputs = [self.fname(x_name), self.dequantize(w_name, w_scale, w_zp, axis=axis)]
        if len(node.input) > 8:  # int32 bias, scale = x_scale * w_scale
            bias = self.to_arr(node.input[8]).astype(np.float64)
            bias_scale = self.to_arr(x_scale).astype(np.float64) * w_scale_arr.astype(np.float64)
            inputs.append(self.add_float_init(node.input[8] + '_f32', bias * bias_scale))
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        self.emit_clipped(helper.make_node(
            'Conv', inputs, list(node.output),
            name=node.name or f'conv_{self.next_uid()}', **attrs), y_scale, y_zp)

    def convert_addend(self, q_name, s_name, zp_name):
        """Return the float operand for one input of QLinearAdd."""
        if q_name in self.inits:  # constant operand: fold to a float initializer
            q_arr = self.to_arr(q_name).astype(np.float64)
            zero = self.to_arr(zp_name).astype(np.float64)
            scale = self.to_arr(s_name).astype(np.float64)
            return self.add_float_init(q_name + '_f32', (q_arr - zero) * scale)
        return self.fname(q_name)

    def convert_node(self, node):
        """Convert one node of the source graph."""
        if node.op_type == 'QuantizeLinear':
            # input quantization: skip, keep the float input
            self.fmap[node.output[0]] = self.fname(node.input[0])
        elif node.op_type == 'DequantizeLinear':
            # output dequantization: skip, keep the float name
            self.fmap[node.output[0]] = self.fname(node.input[0])
        elif node.op_type == 'QLinearConv':
            self.convert_conv(node)
        elif node.op_type == 'QLinearAdd':
            operands = [self.convert_addend(*node.input[idx:idx + 3]) for idx in (0, 3)]
            self.emit_clipped(helper.make_node(
                'Add', operands, list(node.output),
                name=node.name or f'add_{self.next_uid()}'), node.input[6], node.input[7])
        elif node.op_type == 'QLinearGlobalAveragePool':
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            assert attrs.get('channels_last', 0) == 0, 'channels_last unsupported'
            self.emit_clipped(helper.make_node(
                'GlobalAveragePool', [self.fname(node.input[0])], list(node.output),
                name=node.name or f'gap_{self.next_uid()}'), node.input[3], node.input[4])
        elif node.op_type == 'QLinearMatMul':
            weight = self.dequantize(*node.input[3:6])
            self.emit_clipped(helper.make_node(
                'MatMul', [self.fname(node.input[0]), weight], list(node.output),
                name=node.name or f'matmul_{self.next_uid()}'), node.input[6], node.input[7])
        elif node.op_type == 'Reshape':
            self.new_nodes.append(helper.make_node(
                'Reshape', [self.fname(i) for i in node.input], list(node.output),
                name=node.name or f'reshape_{self.next_uid()}'))
            for name in node.input[1:]:
                if name in self.inits:
                    self.keep_init(name)
        else:
            raise ValueError(f'unhandled op: {node.op_type}')

    def convert(self):
        """Convert the whole graph and return the new float-compute model."""
        for node in self.graph.node:
            self.convert_node(node)
        for out in self.graph.output:
            if out.name in self.fmap:
                self.new_nodes.append(helper.make_node(
                    'Identity', [self.fmap[out.name]], [out.name],
                    name=f'out_identity_{out.name}'))
        new_graph = helper.make_graph(
            self.new_nodes, self.graph.name + '_float',
            list(self.graph.input), list(self.graph.output),
            initializer=list(self.new_inits.values()))
        return helper.make_model(
            new_graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=7)


def main():
    """Load the quantized model, convert it, and save the float model."""
    model = onnx.load(SRC)
    new_model = QuantToFloatConverter(model.graph).convert()
    onnx.checker.check_model(new_model)
    onnx.save(new_model, DST)
    print('saved', DST)


if __name__ == '__main__':
    main()
