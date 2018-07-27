# How to run examples

## Build examples (Ubuntu)

Use Cmake.

See [Getting started](getting-started.md) : **Approach 2.** Build with Cmake

- Build example (set your path for NNST_ROOT, then gst plugin path is ```$NNST_ROOT/lib```)

```
# prepare
$ sudo apt-get install python-gi python3-gi  # for python example
$ sudo apt-get install python-gst-1.0 python3-gst-1.0
$ sudo apt-get install python-gst-1.0-dbg python3-gst-1.0-dbg
$ export NNST_ROOT=$HOME/nnstreamer  # set your own path
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NNST_ROOT/lib
$ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$NNST_ROOT/lib
```
```
# build example
$ cd nnstreamer
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -rf *        # Ensure the build directory is empty
$ cmake -DCMAKE_INSTALL_PREFIX=${NNST_ROOT} -DINCLUDE_INSTALL_DIR=${NNST_ROOT}/include ..
$ make install    # Install nnstreamer plugin libraries into $NNST_ROOT/lib
$ cd ..
```

## Example : filter for image classification

```
v4l2src -- tee -- textoverlay -- videoconvert -- ximagesink
            |
            --- videoscale -- tensor_converter -- tensor_filter -- tensor_sink
```

NNStreamer example for image recognition.
- Download tflite moel 'Mobilenet_1.0_224_quant' from [Here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md#image-classification-quantized-models)

Displays video sink.

1. 'tensor_filter' for image recognition. (classification with 224x224 image).
2. 'tensor_sink' updates recognition result to display in textoverlay.

- Run example
```
$ cd build/nnstreamer_example/example_filter
$ ./nnstreamer_example_filter 
```

```
# for python example
$ cd nnstreamer_example/example_filter
$ python nnstreamer_example_filter.py
```

## Example : video mixer with NNStreamer plug-in

```
v4l2src -- tee ------------------------------------------ videomixer -- videoconvert -- ximagesink (Mixed)
            |                                                 |
            --- tensor_converter -- tensordec -- videoscale ---
            |
            --- videoconvert -- ximagesink (Original)
```

Displays two video sinks,

1. Original from cam
2. Mixed : original + scaled (tensor_converter-tensor_dec-videoscale)

In pipeline, converter-decoder passes video frame.

- Run example
```
$ cd build/nnstreamer_example/example_cam
$ ./nnstreamer_example_cam
```

## Example : tensor sink

Two simple examples to use tensor sink.

#### 1. 100 buffers passed to tensor sink

```
videotestsrc -- tensor_converter -- tensor_sink
```

Displays nothing, this sample code shows how to get buffer from tensor sink.

- Run example
```
$ cd build/nnstreamer_example/example_sink
$ ./nnstreamer_sink_example
```

#### 2. launch two pipelines

```
videotestsrc -- tensor_converter -- tensor_sink
[push buffer from tensor_sink to appsrc]
appsrc -- tensordec -- videoconvert -- ximagesink
```

Displays video sink.

Tensor sink receives buffer and pushes it into appsrc in 2nd pipeline.

- Run example
```
$ cd build/nnstreamer_example/example_sink
$ ./nnstreamer_sink_example_play
```
