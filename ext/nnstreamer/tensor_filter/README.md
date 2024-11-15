---
title: NNStreamer tensor\_filter default subplugin manual
...

# List of subplugins

## Language bindings and custom function calls

You can embed arbitrary function calls in a nnstreamer pipeline with these subplugins.

### Custom/C++ (cpp)
### Custom/C (custom)
### Custom/C Simplified (custom-easy)
### Lua
### Python3

## Neural network frameworks

You can insert model files and their frameworks in a nnstreamer pipeline with these subplugins.

### Armnn
### Caffe2
### Pytorch
### Tensorflow
### Tensorflow-lite
- subplugin name: 'tensorflow1-lite'
- subplugin name: 'tensorflow2-lite'
- subplugin name: 'tensorflow2-lite-custom'

#### How to use custom tensorflow-lite binaries

If you want to use tensorflow-lite custom operators with your own tensorflow-lite custom binaries, you can use tensorflow2-lite-custom subplugin. As its name suggests, this supports tensorflow-lite 2.x versions.

By default, this subplugin loads ```./libtensorflow2-lite-custom.so```, which is the user's custom tensorflow-lite binary.

### SNAP
### NCNN
- subplugin name: 'ncnn'
### ExecuTorch
- subplugin name: 'executorch'
### MXNET
### NNFW
### ONNX Runtime
### Openvino
### TVM

## Hardware accelerators (frameworks for specific hardware)


### Vivante (Verisilicon)
- subplugin name: 'vivante'

### Movidius-ncsdk2 (Intel)
### Edgetpu (Google TPU for embedded)
### SNPE (Qualcomm NPU)

Note: due to API disruptions, there are two versions.

### Tensorrt (NVIDIA)

Note: due to API disruptions, there are two versions.

### TRIx-engine (Samsung TV/CE)

## Adaptors for other pipeline frameworks

You may insert pipelines of other frameworks into nnstmreamer pipeline as well.
For example, you can embed a Mediapipe pipeline inside your nnstreamer pipeline.

### Mediapipe
### DALI

## Adaptors for individual usage cases

This could also be implemented as a custom subplugin

### LLAMA2


# How to write a new subplugin

If you want to support a new deep neural network framework or a runtime/backend library of a new hardware accelerator, you need a new tensor-filter subplugin.

Please refer to [/Documentation/writing-subplugin-tensor-filter.md].

