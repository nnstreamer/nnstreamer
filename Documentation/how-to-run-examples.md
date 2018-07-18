# How to run examples

## Build examples (Ubuntu)

Use Cmake.

See [Getting started](getting-started.md) : **Approach 2.** Build with Cmake

- Build example (DEV_ROOT=/dev, and gst plugin path is ```/dev/lib```)

```
# prepare
$ sudo apt-get install python-gi python3-gi  # for python example
$ sudo apt-get install python-gst-1.0 python3-gst-1.0
$ sudo apt-get install python-gst-1.0-dbg python3-gst-1.0-dbg
$ export DEV_ROOT=/dev
$ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$DEV_ROOT/lib
```
```
# build example
$ cd nnstreamer
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -rf *        # Ensure the build directory is empty
$ cmake -DCMAKE_INSTALL_PREFIX=${DEV_ROOT} -DINCLUDE_INSTALL_DIR=${DEV_ROOT}/include ..
$ make install    # Install nnstreamer plugin libraries into /dev/lib
$ cd ..
```

## Example : filter

```
v4l2src -- tee -- textoverlay -- videoconvert -- xvimagesink
            |
            --- tensor_converter -- tensor_filter -- tensor_sink
```

NNStreamer example for image recognition.

Displays video sink.

1. 'tensor_filter' for image recognition.
2. 'tensor_sink' updates recognition result to display in textoverlay.

- Run example
```
# python example
$ cd nnstreamer_example/example_filter
$ python nnstreamer_example_filter.py 
```

```
$ cd build/nnstreamer_example/example_filter
$ ./nnstreamer_example_filter
```

## Example : video mixer

```
v4l2src -- tee ------------------------------------------ videomixer -- videoconvert -- xvimagesink (Mixed)
            |                                                 |
            --- tensor_converter -- tensordec -- videoscale ---
            |
            --- videoconvert -- xvimagesink (Original)
```

Displays two video sinks,
 
1. Original from cam
2. Mixed : original + scaled (tensor_converter-tensor_dec-videoscale)

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
appsrc -- tensordec -- videoconvert -- xvimagesink
```

Displays video sink.

Tensor sink receives buffer and pushes it into 2nd pipeline.
 
- Run example
```
$ cd build/nnstreamer_example/example_sink
$ ./nnstreamer_sink_example_play
```
