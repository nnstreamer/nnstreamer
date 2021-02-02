---
title: Debugging tools
...

## Debugging
 GStreamer has a debugging feature that automatically generates pipeline graphs. 
* https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html

* Table of Contents
  * [Displaying debug messages with $GST_DEBUG](#displaying-debug-messages-with-gst_debug)
  * [Generating pipeline graph with $GST_DEBUG_DUMP_DOT_DIR](#generating-pipeline-graph-with-gst_debug_dump_dot_dir)
  * [Debugging remotely with gst-debugger](#debugging-remotely-with-gstdebugger)


### Displaying debug messages with $GST_DEBUG

If GStreamer has been configured with --enable-gst-debug=yes, this variable can be set to a list of debug options, which cause GStreamer to print out different types of debugging information to stderr. The variable takes a comma-separated list of "category_name:level" pairs to set specific levels for the individual categories. The level value ranges from 0 (nothing) to 9 (MEMDUMP).
For more details, refer to https://gstreamer.freedesktop.org/documentation/tutorials/basic/debugging-tools.html

* 1 (ERROR): Logs all fatal errors.
* 2 (WARNING): Logs all warnings.
* 3 (FIXME): Logs all fixme messages.
* 4 (INFO): Logs all informational messages.
* 5 (DEBUG): Logs all debug messages.
* 6 (LOG): Logs all log messages.
* 7 (TRACE): Logs all trace messages. 
* 9 (MEMDUMP): Log all memory dump messages. 

#### How to use debug options
The category_name can contain "*" as a wildcard. For example, setting GST_DEBUG to GST_AUTOPLUG:6,GST_ELEMENT_*:4, will cause the GST_AUTOPLUG category to be logged at full LOG level, while all categories starting with GST_ELEMENT_ will be logged at INFO level. To get all possible debug output, set GST_DEBUG to *:9. For debugging purposes a *:6 debug log is usually the most useful, as it contains all important information, but hides a lot of noise such as refs/unrefs. For bug reporting purposes, a *:6 log is also what will be requested usually. It's often also worth running with *:3 to see if there are any non-fatal errors or warnings that might be related to the problem at hand. Since GStreamer 1.2 it is also possible to specify debug levels by name, e.g. GST_DEBUG=*:WARNING,*audio*:LOG
```bash
$ export GST_DEBUG=[Level]
```

Use gst-launch-1.0 --gst-debug-help to obtain the list of all registered categories. 
```bash
$ gst-launch-1.0 --gst-debug-help
```

#### Case study 1: Tracing Gstreamer plugins with GST_DEBUG
Traces for buffer flow, events and messages in TRACE level.
```bash
$ GST_DEBUG="GST_TRACER:7,GST_BUFFER*:7,GST_EVENT:7,GST_MESSAGE:7" \
GST_TRACERS=log gst-launch-1.0 fakesrc num-buffers=10 ! fakesink -
```

Print some pipeline stats on exit.
```bash
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="stats;rusage" \
GST_DEBUG_FILE=trace.log gst-launch-1.0 fakesrc num-buffers=10 \
sizetype=fixed ! queue ! fakesink && gst-stats-1.0 trace.log
```

Get ts, average-cpuload, current-cpuload, time, and plot.
* https://github.com/GStreamer/gstreamer/blob/master/scripts/gst-plot-traces.sh
```bash
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="stats;rusage" \
GST_DEBUG_FILE=trace.log /usr/bin/gst-play-1.0 ./your_movie.mp4 && \
./scripts/gst-plot-traces.sh --format=png | gnuplot eog trace.log.*.png
```

Print processing latencies.
```bash
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS=latency gst-launch-1.0  \
audiotestsrc num-buffers=10 ! audioconvert ! volume volume=0.7 ! autoaudiosink
```

Raise a warning if a leak is detected.
```bash
$ GST_TRACERS="leaks" gst-launch-1.0 videotestsrc num-buffers=10 ! fakesink
```

Check if any GstEvent or GstMessage is leaked and raise a warning.
```bash
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="leaks(GstEvent,GstMessage)" \
gst-launch-1.0 videotestsrc num-buffers=10 ! fakesink
```

#### Case study 2: Tracing individual NNStreamer elements with GST_DEBUG and silent property

Each NNStreamer element has the `silent` property. An NNStreamer element can be traced in verbose mode by setting `FALSE` to the `silent` property.

```bash
$ GST_DEBUG="tensor_converter:7" \
gst-launch-1.0 videotestsrc ! tensor_converter silent=FALSE ! tensor_sink
```

In each element's source code, there is `DEFAULT_SILENT` macro that allows you to change the default `silent` value. Setting `FALSE` to `DEFAULT_SILENT` and rebuilding the library will set verbose mode of the element by default without changing your application code.

### Generating pipeline graph with $GST_DEBUG_DUMP_DOT_DIR

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
```

#### How to convert a pipeling dot file to pdf

XDot is an interactive viewer for graphs written in Graphviz's dot language. You can view the pipeline graph graphically with XDot.
```bash
$ sudo apt install xdot
$ xdot 0.00.00.328088000-gst-launch.NULL_READY.dot
```
<img src=gst-debug-dump-dot-dir-xot.png border=0></img>


If you want to get a PDF format file from .dot file, You need to convert the to a graphical format with dot command.
The below exampe shows how to render PAUSED_READY.dot pipeline.
```bash
* Case 1: Convert .dot to .pdf
$ dot -Tpdf 0.00.05.104625000-gst-launch.PAUSED_READY.dot > pipeline_PAUSED_READY.pdf
$ evince pipeline_PaUSED_READY.pdf
* Case 2: Convert .dot to .png
$ dot -Tpng 0.00.05.104625000-gst-launch.PAUSED_READY.dot > pipeline_PAUSED_READY.png
$ eog pipeline_PaUSED_READY.png
```

#### Case study: Test case in NNstreamer

First of all, try to build NNStreamer source code with cmake in nnstreamer/build folder.

```bash
nnstreamer/test$ ./testAll.sh 1
nnstreamer/test$ cd performance/debug/tensor_convertor
$ eog ${number_of_test_case}.png
```

And then, you can see elements and caps graph in a pipeline.


### Debugging remotely with gst-debugger
gst-debugger (a.k.a Gstreamer Debugger) toolset allows to introspect gst-pipeline remotely. It provides graphical client, and GStreamer's plugin.
This guide is written on Ubuntu 16.04 X86_64 distribution.

#### Build the source code
```bash
$ git clone https://github.com/GNOME/gst-debugger.git
$ cd gst-debugger
$ git checkout 0.90.0
$ ./autogen.sh
$ vi ./src/gst-debugger/controller/controller.cpp
---------------- patch: start --------------------
@@ -16,6 +16,7 @@
 #include "common/common.h"

 #include <gtkmm.h>
+#include <iostream>

 Controller::Controller(IMainView *view)
  : view(view)
---------------- patch: end ----------------------
$ make -j`nproc`
$ sudo make install
$ ls -al /usr/local/lib/libgst-debugger-common*.so.*
lrwxrwxrwx 1 root root      37 Nov 19 14:59 /usr/local/lib/libgst-debugger-common-c-0.1.so.0 -> libgst-debugger-common-c-0.1.so.0.0.0
-rwxr-xr-x 1 root root  250808 Nov 19 14:59 /usr/local/lib/libgst-debugger-common-c-0.1.so.0.0.0
lrwxrwxrwx 1 root root      39 Nov 19 14:59 /usr/local/lib/libgst-debugger-common-cpp-0.1.so.0 -> libgst-debugger-common-cpp-0.1.so.0.0.0
-rwxr-xr-x 1 root root 1848904 Nov 19 14:59 /usr/local/lib/libgst-debugger-common-cpp-0.1.so.0.0.0

```

#### Run gst-debugger
The toolset consists of the rich client, and debugserver (Default port: 8080). debugserver is implemented as a tracer plugin, and has to be loaded with your pipeline as following:
```bash
Case1: Run videotestsrc plugin with gst-launch command.
$ GST_TRACERS="debugserver(port=8080)" gst-launch-1.0 videotestsrc ! autovideosink

Case2: Run a Totem movie player that is developed by GStreamer.
$ wget http://www.html5videoplayer.net/videos/toystory.mp4
$ GST_TRACERS="debugserver(port=8080)" totem  ./toystory.mp4
```

Now you can use a debugging client to connect to the debugger and inspect your pipeline as following:
```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
$ gst-debugger-1.0
```
Starting from now, enjoy a debugging with gst-debugger.
<img src=gstreamer-debugger-screen.png border=0></img>
