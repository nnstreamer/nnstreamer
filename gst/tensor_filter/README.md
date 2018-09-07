# NNStreamer::tensor\_filter

This is the main element of the whole NNStreamer project. This connects gstreamer data stream with neural network frameworks (NNFW) such as Tensorflow or Caffe. ```tensor_filter``` is supposed to attach an instance of neural network model with the given NNFW as a filter to a gstreamer stream. The input/output stream data type is either ```other/tensor``` or ```other/tensors```.

With ```other/tensor```, you may push (or pull) a single tensor for an instance of inference for the given model.

With ```other/tensors```, you may push (or pull) multiple tensors for an instance of inference for the given model. For example, if the output of a neural network model has two distinctive output tensors, "Bounding boxes, uint32[100][4]" and "Labels, uint32[100]", you may use ```other/tensors``` as the source pad capability to have both tensors in a single frame of the source pad. Note that ```tensor_filter``` supports one always source pad and one always sink pad.

# Supported Features

- Multi-tensor (experimental. from 0.0.2+)
- Tensorflow-lite (stable. from 0.0.1)
- Custom filters (stable. from 0.0.1)

# Planned Features

- Framerate policies (for 0.0.2)
- Timestamp handling (for 0.0.2)
- Recurrent network support (for 0.0.2)
- Tensorflow (later than 0.0.2)
- Caffe/Caffe2 (later than 0.0.2)

# Known Bugs or Concerns

- Multi-tensor input/output is not throughly tested.

# Sink Pads

One always sink pad exists. The capability of sink pad is ```other/tensor``` and ```other/tensors```.

The number of frames in a buffer is always 1. Althugh the data semantics of a tensor may have multiple distinct data frames in a single tensor.

# Source Pads

One always source pad exists. The capability of source pad is ```other/tensor``` and ```other/tensors```.

The number of frames in a buffer is always 1. Althugh the data semantics of a tensor may have multiple distinct data frames in a single tensor.

# Performance Characteristics

- We do not support in-place operations with tensor\_filter. Actually, with tensor\_filter, in-place operations are considered harmful for the performance and correctness.
- It is supposed that There is no memcpy from the previous element's source pad to this element's sink or from this element's source to the next element's sink pad.
    - This is something we need to verify later (later than 0.0.2).

# Details

## Sub-Components

### Main ```tensor_filter.c```

This is the main placeholder for all different subcomponents. With the property, ```FRAMEWORK```, this main component loads the proper subcomponent (e.g., tensorflow-lite support, custom support, or other addtional NNFW supports).

The main component is supposed process the standard properties for subcomponents as well as processing the input/output dimensions.

The subcomponents as supposed to fill in ```GstTensor_Filter_Framework``` struct and register it with ```supported``` array in ```tensor_filter.h```.

Note that the registering sturcture may be updated later. (We may follow what ```Linux.kernel/drivers/devfreq/devfreq.c``` does)

### Tensorflow-lite support, ```tensor_filter_tensorflow_lite.c```

This should fill in ```GstTensor_Filter_Framework``` supporting tensorflow_lite.

### Custom function support, ```tensor_filter_custom.c```

Neural network and streameline developers may define their own tensor postprocessing operations with tensor_filter_custom.

With ```nnstreamer-devel``` package installed at build time (e.g., ```BuildRequires: pkgconfig(nnstreamer)``` in .spec file), develerops can implement their own functions and expose their functions via ```NNStreamer_custom_class``` defined in ```tensor_fitler_custom.h```. The resulting custom developer plugin should exist as a shared library (.so) with the symbol NNStreamer_custom exposed with all the func defined in NNStreamer_custom_class.

@TODO Write an example custom filter for novice developers.

### We may add other NNFW as well (tensorflow, caffe, ...)

