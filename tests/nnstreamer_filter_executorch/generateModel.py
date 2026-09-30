##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    generateModel.py
# @brief   Regenerate the .pte fixtures runTest.sh runs the executorch filter on
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# The models are committed, so this only runs when a fixture has to be rebuilt.
# That happened once already. Program::load takes the constant segment path
# only when constant_segment.offsets is non-empty and otherwise falls back on
# the inline constant_buffer, which ExecuTorch 1.x compiles out
# (ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0) and answers with
# Error::InvalidProgram. The 2024 exporter wrote a constant segment only when
# there was something to put in it, so the model with no constants at all took
# that path - which is why only one of the two fixtures had to be replaced.
#
# Needs the ExecuTorch python package and the torch release it pins:
#   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install executorch==1.4.1
#
# The non-float32 fixtures pin the tensor type mapping of the filter: models
# covering every type it accepts, plus a bfloat16 model it has to refuse since
# NNStreamer has no bfloat16 type. A PT2E quantized model keeps float32 at its
# boundary; integer boundaries come from integer inputs such as token ids, or
# from folding the boundary quantize/dequantize ops into the I/O with the
# QuantizeInputs/QuantizeOutputs passes in executorch.exir.passes.
#
# Usage: python3 generateModel.py [output-directory]   (in this directory)

"""Regenerate the ExecuTorch .pte fixtures used by runTest.sh."""

import os
import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class TwoInputTwoOutput(torch.nn.Module):
    """Adds a different constant to each of two inputs."""

    def forward(self, x, y):
        """Return the two inputs offset by 1.0 and 2.0."""
        return x + 1.0, y + 2.0


class TwoInputOneOutput(torch.nn.Module):
    """Adds the two inputs together."""

    def forward(self, x, y):
        """Return the sum of the two inputs.

        runTest.sh feeds both inputs from one tee, so this doubles the input,
        which is what generateTest.py writes as the golden.
        """
        return x + y


class AddOne(torch.nn.Module):
    """Adds one to its input, in the input's own type."""

    def forward(self, x):
        """Return the input plus one."""
        return x + 1


class MultiType(torch.nn.Module):
    """Adds one to each of six numeric inputs and negates a seventh, bool input."""

    def forward(self, u8, i8, i16, i32, i64, f64, b):
        """Return every input plus one, and the logical not of the bool input."""
        return u8 + 1, i8 + 1, i16 + 1, i32 + 1, i64 + 1, f64 + 1, torch.logical_not(b)


class SumToBfloat16(torch.nn.Module):
    """Sums its float32 inputs into a bfloat16 output."""

    def forward(self, *inputs):
        """Return the sum of every input, cast to bfloat16."""
        return torch.stack(inputs).sum(0).to(torch.bfloat16)


def save_model(path, model, example_args):
    """Export model to the ExecuTorch program format and write it to path."""
    program = to_edge_transform_and_lower(export(model.eval(), example_args)).to_executorch()
    with open(path, 'wb') as file:
        file.write(program.buffer)
    print(f'wrote {path} ({os.path.getsize(path)} bytes)')


def main():
    """Write every fixture into the directory given on the command line."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '../test_models/models'

    save_model(os.path.join(out_dir, 'sample_3x4_two_input_two_output.pte'),
               TwoInputTwoOutput(), (torch.rand(3, 4), torch.rand(3, 4)))
    save_model(os.path.join(out_dir, 'sample_4x4x4x4x4_two_input_one_output.pte'),
               TwoInputOneOutput(), (torch.rand(4, 4, 4, 4, 4), torch.rand(4, 4, 4, 4, 4)))

    save_model(os.path.join(out_dir, 'sample_3x4_multi_type.pte'), MultiType(),
               tuple(torch.zeros(3, 4, dtype=t) for t in
                     (torch.uint8, torch.int8, torch.int16, torch.int32,
                      torch.int64, torch.float64, torch.bool)))
    save_model(os.path.join(out_dir, 'sample_3x4_uint8_add_one.pte'),
               AddOne(), (torch.zeros(3, 4, dtype=torch.uint8),))
    save_model(os.path.join(out_dir, 'sample_3x4_float16_add_one.pte'),
               AddOne(), (torch.zeros(3, 4, dtype=torch.float16),))
    save_model(os.path.join(out_dir, 'sample_3x4_bfloat16_add_one.pte'),
               AddOne(), (torch.zeros(3, 4, dtype=torch.bfloat16),))
    # 17 inputs, so that the input info has spilled past the 16 tensors kept
    # inline by the time the output type is refused.
    save_model(os.path.join(out_dir, 'sample_17_input_bfloat16_output.pte'),
               SumToBfloat16(), tuple(torch.zeros(3, 4) for _ in range(17)))


if __name__ == '__main__':
    main()
