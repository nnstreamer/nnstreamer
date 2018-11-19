
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


### gst-debugger
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
