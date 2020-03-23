#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2019 Samsung Electronics
# License: LGPL-2.1
#
# @file checkLabel.py
# @brief Check the result label of pytorch model
# @author Parichay Kapoor <pk.kapoor@samsung.com>

import torch
import torch.nn as nn
import os

pytorch_save_file = "mnist_cnn.pt"

##
# This code creates a wrapper around mnist model provided by pytorch.
# The wrapper consists of preprocessing as well as converting to appropriate
# type and format as a postprocessing step.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def mnist_loaded():
    mnist_model = Net()
    if (not os.path.isfile(pytorch_save_file)):
        print("Base pytorch save file is missing")
        print("refer https://github.com/pytorch/examples/tree/master/mnist")
    mnist_model.load_state_dict(torch.load(pytorch_save_file))
    mnist_model = mnist_model.to("cpu")
    return mnist_model

class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([0.1307])
                                        .resize_(1, 1, 1, 1))
        self.std_dev = torch.nn.Parameter(torch.tensor([0.3081])
                                        .resize_(1, 1, 1, 1))
        self.softmax = torch.nn.Softmax(dim=1)

        mnist_model = mnist_loaded()
        mnist_model.eval()
        self.mnist = torch.jit.trace(mnist_model,
                                      torch.rand(1, 1, 28, 28))

    @torch.jit.script_method
    def helper(self, input):
      float_input = input.float() / 255
      float_input = float_input.transpose_(2,3).transpose_(1,2)
      float_output = self.mnist((float_input - self.means)/self.std_dev)
      return (self.softmax(float_output) * 255).byte()

    @torch.jit.script_method
    def forward(self, input):
        return self.helper(input)

model = MyScriptModule()

traced_script_module = torch.jit.trace(model, torch.ones(size=(1,28,28,1)))
traced_script_module.save("pytorch_lenet5.pt")

# This is testing code to verify that the generated model file is working correctly
from PIL import Image
import numpy as np

image_file = 'img/9.png'
img_pil = Image.open(image_file)
img_pil = Image.composite(img_pil, Image.new('L', img_pil.size, 'white'), img_pil)
img_np = np.array(img_pil).astype(np.uint8).reshape(1,28,28,1)

out = traced_script_module(torch.ones(1,28,28,1)).data.numpy()
out = traced_script_module(torch.Tensor(img_np)).data.numpy()
print(out.argmax())
