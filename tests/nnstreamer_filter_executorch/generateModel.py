#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file generateModel.py
# @brief Regenerate the .pte fixtures used by runTest.sh
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
#
# The models are committed, so this is only run when a fixture has to be
# rebuilt - for instance after an ExecuTorch release stops accepting the
# serialized form the previous one produced.
#
# Needs the ExecuTorch python package and the torch release it pins:
#   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install executorch==1.4.1
#
# usage: ./generateModel.py [output-directory]

import os
import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from torch.export import export


class TwoInputTwoOutput(torch.nn.Module):
    """@brief Adds a different constant to each of two inputs."""

    def forward(self, x, y):
        return x + 1.0, y + 2.0


class TwoInputOneOutput(torch.nn.Module):
    """@brief Adds the two inputs together.

    runTest.sh feeds both inputs from one tee, so the output is the input
    doubled - which is exactly the golden generateTest.py writes.
    """

    def forward(self, x, y):
        return x + y


def save_model(path, model, example_args):
    program = to_edge_transform_and_lower(export(model.eval(), example_args)).to_executorch()
    with open(path, 'wb') as file:
        file.write(program.buffer)
    print('wrote {} ({} bytes)'.format(path, os.path.getsize(path)))


out_dir = sys.argv[1] if len(sys.argv) > 1 else '../test_models/models'

save_model(os.path.join(out_dir, 'sample_3x4_two_input_two_output.pte'),
           TwoInputTwoOutput(), (torch.rand(3, 4), torch.rand(3, 4)))
save_model(os.path.join(out_dir, 'sample_4x4x4x4x4_two_input_one_output.pte'),
           TwoInputOneOutput(), (torch.rand(4, 4, 4, 4, 4), torch.rand(4, 4, 4, 4, 4)))
