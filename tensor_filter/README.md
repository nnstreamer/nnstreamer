# NNStreamer::tensor\_filter

## Motivation

This is the main element of the whole NNStreamer suite. This connects gstreamer data stream with neural network frameworks (NNFW) such as Tensorflow or Caffe. ```tensor_filter``` is supposed to attach an instance of neural network model with the given NNFW as a filter to a gstreamer stream. The input/output stream data type is ```other/tensor```.

## Components

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

