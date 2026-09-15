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
- subplugin name: 'python3'

The model is a python script defining the class ```CustomFilter``` with either ```getInputDim()``` and ```getOutputDim()```, or ```setInputDim()```, and ```invoke(input_array)```. ```tests/test_models/models/passthrough.py``` is an example. The custom property is split by spaces into the arguments of the class constructor.

```invoke()``` gets a list of 1-D numpy arrays, one for each input tensor, and returns a list of numpy arrays, one for each output tensor, each with the type and the byte size of its tensor. An output array is copied before it is pushed if it is not contiguous, if its data lies in an input tensor, or if its data pointer is the same as that of an output already given out. Any other array is pushed without a copy, including a read-only array and a view overlapping another output, so an element after tensor\_filter must not write into such an output. Any other result, or an exception, fails the invoke and the stream stops with an error.

## Neural network frameworks

You can insert model files and their frameworks in a nnstreamer pipeline with these subplugins.

### Armnn
### Caffe2
### Pytorch
### Tensorflow
### Tensorflow-lite
- subplugin name: 'tensorflow1-lite'
- subplugin name: 'tensorflow2-lite'

The former ```tensorflow2-lite-custom``` subplugin, which used a custom tensorflow-lite binary, is removed (#4268); use ```tensorflow2-lite``` instead. Its meson option ```tflite2-custom-support``` is kept only so that existing build scripts passing it keep configuring. The option selects nothing, and ```-Dtflite2-custom-support=enabled``` is rejected at ```meson setup```. The option itself will be removed later (#4978), so drop it from build scripts.

### LiteRT
- subplugin name: 'litert'

This subplugin targets the LiteRT 2.x "CompiledModel" C API, the successor line of TensorFlow-Lite. It consumes the standard ```.tflite``` flatbuffer model and may coexist with the classic-Interpreter-API ```tensorflow2-lite``` subplugin, so the two runtimes can be compared on the same model in one pipeline.

Custom properties (```custom=Key1:Value1,Key2:Value2```):
- ```Accelerators```: hardware accelerators to enable: cpu, gpu, npu; combinable with '+' (e.g., ```Accelerators:npu+cpu```). Default: cpu. The standard ```accelerator``` property is also honored.
- ```Signature```: the key of the model signature to run. Default: the first signature.

To build, provide the LiteRT SDK via pkg-config (```litert.pc```) or the ```LITERT_ROOT``` environment variable (expecting ```$LITERT_ROOT/lib/libLiteRt.so``` and ```$LITERT_ROOT/include/litert/c/*.h```), and use the ```litert-support``` meson option. The "install LiteRT SDK" steps in ```.github/workflows/ubuntu_clean_meson_build.yml``` and ```.github/workflows/macos.yaml``` are working examples that assemble the SDK from the official LiteRT releases (headers from ```litert_cc_sdk.zip```, the runtime library from the ```ai-edge-litert``` wheel).

### SNAP
### NCNN
- subplugin name: 'ncnn'
### ExecuTorch
- subplugin name: 'executorch'

This subplugin runs PyTorch models exported to the ExecuTorch ```.pte``` format, through the ExecuTorch ```Module``` API.

ExecuTorch has no distro package and publishes no prebuilt binaries for desktop Linux or macOS; its pip wheel carries headers but no runtime library. Build and install the runtime with ```tools/executorch-install.sh [install-prefix] [git-tag]``` first, then point ```PKG_CONFIG_PATH``` at ```<install-prefix>/lib/pkgconfig``` and use the ```executorch-support``` meson option. ```.github/workflows/executorch.yml``` is a working example.

That script exists because the pkg-config file ExecuTorch generates links the runtime alone, without the operator kernels; a subplugin built against the stock file links cleanly and then fails on the first invoke with an unregistered operator.

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

