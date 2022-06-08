---
title: NNStreamer tensor\_filter default subplugin manual
...

# Introduction

## What is tensor-filter subplugin?

(( TBU ))


## Which subplugins are located here?

(( TBU ))

# List of subplugins

## Vivante
- subplugin name: 'vivante'

## Movidius-ncsdk2
## NNFW
## Armnn
## Caffe2
## CPP (C++)
## Edgetpu
## Lua
## Mediapipe
## Openvino
## Python3
## Pytorch
## Snap
## SNPE
## Tensorflow
## Tensorflow-lite
- subplugin name: 'tensorflow1-lite'
- subplugin name: 'tensorflow2-lite'
- subplugin name: 'tensorflow2-lite-custom'

### How to use custom tensorflow-lite binaries

If you want to use tensorflow-lite custom operators with your own tensorflow-lite custom binaries, you can use tensorflow2-lite-custom subplugin. As its name suggests, this supports tensorflow-lite 2.x versions.

By default, this subplugin loads ```./libtensorflow2-lite-custom.so```, which is the user's custom tensorflow-lite binary.

## Tensorrt
## TRIx-engine
## TVM


# How to write a new subplugin

If you want to support a new deep neural network framework or a runtime/backend library of a new hardware accelerator, you need a new tensor-filter subplugin.

## Write a subplugin in C

Reference (example): [/ext/nnstreamer/tensor_filter/tensor_filter_nnfw.c]
Interface: [/gst/nnstreamer/include/nnstreamer_plugin_api_filter.h]


If the framework or backend/runtime library has C APIs and you want to write the subplugin in C, use ```#include <nnstreamer_plugin_api_filter.h>```.
Your C subplugin is supposed to fill in ```GstTensorFilterFramework``` struct and register the struct with ```nnstreamer_filter_probe (GstTensorFilterFrameworkEventData *)``` function, which is supposed to be called with ```((constructor))``` initializer (```init_filter_nnfw (void)``` function in the reference).
If your subplugin has custom properties to be supplied by users, describe their usages with ```nnstreamer_filter_set_custom_property_desc ()``` function.
Then, call ```nnstreamer_filter_exit ()``` function with ```((desctructor))``` terminator (```fini_filter_nnfw (void)``` function in the reference).


In ```GstTensorFilterFramework```, there are two different ways, ```v0 (version == GST_TENSOR_FILTER_FRAMEWORK_V0)``` and ```v1 (version == GST_TENSOR_FILTER_FRAMEWORK_V1)```. In the struct, there is a ```union``` of ```v0``` and ```v1```, and it is recommended to use ```v1``` and ```set version = GST_TENSOR_FILTER_FRAMEWORK_V1``` (v1). ```v0``` is supposed to be used by old subplugins for backward compatibilty and any new subplugins should use ```v1```, which is simpler and richers in features.


However, note that if you are going to use framework/library with C++ APIs, please do not use ```nnstreamer_plugin_api_filter.h```, but use the base tensor-filter-subplugin C++ class as in the next section.

## Write a subplugin in C++


Reference (example): [/ext/nnstreamer/tensor_filter/tensor_filter_snap.cc]
Interface: [/gst/nnstreamer/include/nnstreamer_cppplugin_api_filter.hh]


If the framework or backend/runtime library has C++ APIs or you want to write the subplugin in C++, use ```#include <nnstreamer_cppplugin_api_filter.hh>``` and inherit the base class, ```nnstreamer::tensor_filter_subplugin```.
With this interface, subplugin writers are supposed to write their own concrete class based on ```nnstreamer::tensor_filter_subplugin``` by filling up virtual methods (methods marked ```These should be filled/implemented by subplugin authores```).
Mandatory methods to be filled are declared as pure virtual function and optional methods are declared as regular virtual function (```eventHandler```).


As in C subplugin, the derived (concrete) class should be registered at init and unregistered at exit.
Subplugin writers are supposed to use the static methods of the base class, ```register_subplugin()``` and ```unregister_subplugin()```; refer to the function ```init_filter_snap()``` and ```fini_filter_snap()``` in the reference example.



Note that C++ subplugin is simpler and easy-to-maintain compared to C subplugin. Unless you really need to write your subplugin in C, we recommend to use the ```nnstreamer::tensor_filter_subplugin``` base class.
