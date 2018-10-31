

Table of Contents
=================

* [Tools](#tools)
  * [Development](##development)
  * [Tracing](##tracing)
  * [Debugging](##debugging)
  * [Profiling](##profiling)


# Tools

This section describes how to do tracing, debugging, profiling when developers deploy NNStreamer on their own devices. 
There are three features as following: 

- Development
- Tracing(TBD)
- Debugging
- Profiling(in progress)

If you are interested in the tool technolog to optimize the your application using NNStremaer, Please refer to the below issue. 
* https://github.com/nnsuite/nnstreamer/issues/132


## Development 
* getTestModels.sh: Get network model for evaluation
* gst-indent: Check for existence of indent, and error out if not present
* pre-commit: Verify what is about to be committed

## Tracing
* WIP


## Debugging

### Using $GST_DEBUG_DUMP_DOT_DIR
* https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html#getting-pipeline-graphs

To obtain .dot files, simply set the GST_DEBUG_DUMP_DOT_DIR environment variable to point to the folder where you want the files to be placed.

#### Prerequisite
You must install the below packages to visualize the pipeline operations.
```bash
$ sudo apt install eog
$ sudo apt install graphviz
$ sudo apt install gsteamer1.0-tools
```

#### How to run

First of all, try to build NNStreamer source code with cmake in nnstreamer/build folder.

```bash
nnstreamer/test$ ./testAll.sh 1
nnstreamer/test$ cd performance/debug/tensor_convertor
$ eog ${number_of_test_case}.png
```

then, you can see elements and caps in pipeline.

## Profiling

### Using $GST_DEBUG_DUMP_TRACE_DIR
The gst-instruments tool is an easy-to-use profiler for GStreamer.
* https://github.com/kirushyk/gst-instruments.git
* gst-instruments displays the trace file.
* gst-top is inspired by top and perf-top, this utility displays performance report for the particular command, analyzing GStreamer ABI calls.

#### Prerequisite

- autoreconf
- pkg-config
- automake
- libtool
- gst-instruments

```bash
$ git clone https://github.com/kirushyk/gst-instruments.git
$ cd gst-instruments
$ ./autogen.sh
$ make
$ sudo make install
```

#### How to run

```bash
nnstreamer/test$ ./testAll.sh 1
nnstreamer/test$ cd performance/profile/tensor_convertor
$ eog ${number_of_test_case}.svg
```

Then, you can see a time cost and a CPU usage in pipeline.
