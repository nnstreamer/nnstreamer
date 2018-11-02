

Table of Contents
=================

* [Tools](#tools)
  * [Development](#development)
  * [Tracing](#tracing)
  * [Debugging](#debugging)
  * [Profiling](#profiling)


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
 GStreamer has a debugging feature that automatically generates pipeline graphs. 
* https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html

### Using $GST_DEBUG_DUMP_DOT_DIR

Regardless of whether you are using gst-launch-1.0 or a GStreamer application, you have to need to define the GST_DEBUG_DUMP_DOT_DIR environment variable.
GStreamer uses this environment variable as the output location to generate pipeline graphs.

To obtain .dot files, simply set the GST_DEBUG_DUMP_DOT_DIR environment variable to point to the folder where you want the files to be placed.
* https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html#getting-pipeline-graphs

#### Prerequisite
You must install the below packages to visualize the pipeline operations.
```bash
$ sudo apt -y install graphviz libgstreamer1.0-dev
$ sudo apt -y install xdot gsteamer1.0-tools
```

#### Gstreamer application macros for custom GSteamer application

If you're using a custom GStreamer application, you'll need to use GStreamer debug macros to trigger pipeline generation.
For instance, to see a complete pipeline graph, add the following macro invocation at the point in your application where your pipeline elements have been created and linked:
* [GST_DEBUG_BIN_TO_DOT_FILE()](https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-GstInfo.html#GST-DEBUG-BIN-TO-DOT-FILE:CAPS)
* [GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS()](https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer/html/gstreamer-GstInfo.html#GST-DEBUG-BIN-TO-DOT-FILE-WITH-TS:CAPS)

```bash
GST_DEBUG_BIN_TO_DOT_FILE(pipeline, GST_DEBUG_GRAPH_SHOW_ALL, "pipeline")
```
If you are using a custom GStreamer app, pipeline files will only be triggered based on your invocation of the GST_DEBUG_BIN_TO_DOT_FILE() macros.

#### How to run

```bash
$ export GST_DEBUG_DUMP_DOT_DIR=./tracing/
$ mkdir tracing
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="framerate" gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink sync=true
$ ls -al ./tracing/
  0.00.00.328088000-gst-launch.NULL_READY.dot
  0.00.00.330350000-gst-launch.READY_PAUSED.dot
  0.00.02.007860000-gst-launch.PAUSED_PLAYING.dot
  0.00.05.095596000-gst-launch.PLAYING_PAUSED.dot
  0.00.05.104625000-gst-launch.PAUSED_READY.dot
$ xdot 0.00.00.328088000-gst-launch.NULL_READY.dot
```

#### Case study: NNstreamer's test case

First of all, try to build NNStreamer source code with cmake in nnstreamer/build folder.

```bash
nnstreamer/test$ ./testAll.sh 1
nnstreamer/test$ cd performance/debug/tensor_convertor
$ eog ${number_of_test_case}.png
```

And then, you can see elements and caps graph in a pipeline.


### Using GstShark
[GstShark](https://developer.ridgerun.com/wiki/index.php?title=GstShark) is an open-source project from Ridgerun that provides benchmarks and profiling tools for GStreamer 1.7.1 (and above).
It includes tracers for generating debug information plus some tools to analyze the debug information.
GstShark provides easy to use and useful tracers, paired with analysis tools to enable straightforward optimizations.
GstShark leverages GStreamer's tracing hooks and open-source and standard tracing and plotting tools to simplify the process of understanding the bottlenecks in your pipeline.

```bash
$ sudo apt install libgstreamer1.0-dev
$ sudo apt install graphviz libgraphviz-dev
$ $ sudo apt install octave epstool babeltrace
$ git clone https://github.com/RidgeRun/gst-shark/
$ cd gst-shark
$ ./autogen.sh
$ make
$ sudo make install
```

#### Tracers of GstShark
* InterLatency:	Measures the latency time at different points in the pipeline.
* ProcTime:	Measures the time an element takes to produce an output given the corresponding input.
* Framerate:	Measures the amount of frames that go through a src pad every second.
* ScheduleTime:	Measures the amount of time between two consecutive buffers in a sink pad. T
* CPUUsage:	Measures the CPU usage every second. In multiprocessor systems this measurements are presented per core.
* Graphic:	Records a graphical representation of the current pipeline.
* Bitrate:	Measures the current stream bitrate in bits per second.
* Queue Level:	Measures the amount of data queued in every queue element in the pipeline.
* Buffer:	Prints information of every buffer that passes through every sink pad in the pipeline. This information contains PTS and DTS, duration, size, flags and even refcount.

#### Generating  trace files
The user has the capability of selecting the tracers that will be run with the pipeline by listing them (separated by the character ";") using the option "GST_TRACER_PLUGINS" or "GST_TRACERS", depending on the GStreamer version, at the time of running the pipeline in a very similar way than the "GST_DEBUG" option.
```bash
$ export GST_SHARK_LOCATION=./profile/
# For GStreamer 1.7.1
$ GST_DEBUG="GST_TRACER:7" GST_TRACER_PLUGINS="cpuusage;proctime;framerate"\
     gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink
# For GStreamer 1.8.1 and later
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="cpuusage;proctime;framerate"\
     gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink
```

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
