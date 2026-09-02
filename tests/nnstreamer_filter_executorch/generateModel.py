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
# To extend this into quantized or delegated fixtures - the filter is still
# float32 only, so a quantized fixture is a prerequisite for widening it -
# quantize the exported program with prepare_pt2e/convert_pt2e from
# torch.ao.quantization.quantize_pt2e before to_edge_transform_and_lower, and
# pass a partitioner to that call to delegate to a backend.
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


def save_model(path, model, example_args):
    """Export model to the ExecuTorch program format and write it to path."""
    program = to_edge_transform_and_lower(export(model.eval(), example_args)).to_executorch()
    with open(path, 'wb') as file:
        file.write(program.buffer)
    print(f'wrote {path} ({os.path.getsize(path)} bytes)')


def main():
    """Write both fixtures into the directory given on the command line."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '../test_models/models'

    save_model(os.path.join(out_dir, 'sample_3x4_two_input_two_output.pte'),
               TwoInputTwoOutput(), (torch.rand(3, 4), torch.rand(3, 4)))
    save_model(os.path.join(out_dir, 'sample_4x4x4x4x4_two_input_one_output.pte'),
               TwoInputOneOutput(), (torch.rand(4, 4, 4, 4, 4), torch.rand(4, 4, 4, 4, 4)))


if __name__ == '__main__':
    main()
