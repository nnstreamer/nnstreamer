# How to run examples

## Build examples (Ubuntu)

Use Cmake.

See [Getting started](getting-started.md) : **Approach 2.** Build with Cmake

Build example (DEV_ROOT=/dev, and gst plugin path is ```/dev/lib```)

```
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -rf *        # Ensure the build directory is empty
$ cmake -DCMAKE_INSTALL_PREFIX=${DEV_ROOT} -DINCLUDE_INSTALL_DIR=${DEV_ROOT}/include ..
$ make install    # Install nnstreamer plugin libraries into /dev/lib
$ cd ..
``` 

## Example : cam

```
v4l2src -- tee ------------------------------------------ videomixer -- videoconvert -- xvimagesink
            |                                                 |
            --- tensor_converter -- tensordec -- videoscale ---
            |
            --- videoconvert -- xvimagesink
```

Displays two video sinks,
 
1. Original from cam (640 x 480)
2. Mixed : original + scaled (tensor_converter-tensor_dec-videoscale)

```
$ cd build/nnstreamer_example/example_cam
$ ./nnstreamer_example_cam --gst-plugin-path=/dev/lib 
```

## Example : tensor sink

Two simple examples to use tensor sink.

#### 1. 100 buffers passed to tensor sink

```
videotestsrc -- tensor_converter -- tensor_sink
```

Displays nothing, this sample code shows how to get buffer from tensor sink.

```
$ cd build/nnstreamer_example/example_sink
$ ./nnstreamer_sink_example --gst-plugin-path=/dev/lib 
```

#### 2. launch two pipelines

```
videotestsrc -- tensor_converter -- tensor_sink
[push buffer from tensor_sink to appsrc]
appsrc -- tensordec -- videoconvert -- xvimagesink
```

Displays video sink.

Tensor_sink receives buffer and pushes it into 2nd pipeline.
 
```
$ cd build/nnstreamer_example/example_sink
$ ./nnstreamer_sink_example_play --gst-plugin-path=/dev/lib 
```
