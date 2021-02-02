---
title: Profiling tools
...

## Profiling

### NNShark

Press [here](https://github.com/nnstreamer/nnshark) for further information.

### gst-instruments
The [gst-instruments](https://github.com/kirushyk/gst-instruments) tool is an easy-to-use profiler for GStreamer.
This guide is experimented on Ubuntu 16.04 x86_64 distribution.
* gst-top displays performance report in real time such as top and perf-top utility.
* gst-report-1.0 generates the trace file the below two goals as following:
   * It display CPU usage, time usage, and execution time amon the elements. We can easily profile who spends CPU resource mostly. 
   * It generate a performance graph from a GStreamer pipeline. The graph shows a transfer size as well as CPU usage, time usage, and execution time.

#### Prerequisite
```bash
$ sudo apt install autoreconf pkg-config automake libtool
```

#### How to build source code
```bash
$ git clone https://github.com/kirushyk/gst-instruments.git
$ cd gst-instruments
$ git checkout -v 0.2.3 0.2.3
$ vi ./autogen.sh
--------------- patch: start ----------------------------------
@@ -19,7 +19,7 @@ which "aclocal" 2>/dev/null || {
 }

 echo "Checking for libtool..."
-which "libtool" 2>/dev/null || {
+which "libtoolize" 2>/dev/null || {
   echo "Please install libtool."
   exit 1
 }
--------------- patch: end   ----------------------------------
$ ./autodgen.sh
$ ./configure
$ make -j`nproc`
$ sudo make install 
$ ls /usr/local/lib/libgstintercept.* -al
-rw-r--r-- 1 root root 232492 Nov 21 11:49 /usr/local/lib/libgstintercept.a
-rwxr-xr-x 1 root root   1039 Nov 21 11:49 /usr/local/lib/libgstintercept.la
lrwxrwxrwx 1 root root     24 Nov 21 11:49 /usr/local/lib/libgstintercept.so -> libgstintercept.so.0.0.0
lrwxrwxrwx 1 root root     24 Nov 21 11:49 /usr/local/lib/libgstintercept.so.0 -> libgstintercept.so.0.0.0
-rwxr-xr-x 1 root root 121312 Nov 21 11:49 /usr/local/lib/libgstintercept.so.0.0.0
```

#### gst-top
```bash
$ gst-top-1.0 gst-launch-1.0 audiotestsrc num-buffers=1000 ! vorbisenc ! vorbisdec ! fakesink
Setting pipeline to PAUSED ...
Pipeline is PREROLLING ...
Redistribute latency...
Pipeline is PREROLLED ...
Setting pipeline to PLAYING ...
New clock: GstSystemClock
Got EOS from element "pipeline0".
Execution ended after 0:00:00.301137359
Setting pipeline to PAUSED ...
Setting pipeline to READY ...
Setting pipeline to NULL ...
Freeing pipeline ...
ELEMENT        %CPU   %TIME   TIME
vorbisenc0      60.1   86.9    204 ms
vorbisdec0       8.0   11.6   27.3 ms
fakesink0        1.0    1.5   3.48 ms
audiotestsrc0    0.0    0.0      0 ns
pipeline0        0.0    0.0      0 ns
```

#### gst-report

##### How to display a performanc report with a trace file
```bash
$ LD_PRELOAD=/usr/local/lib/libgstintercept.so \
    GST_DEBUG_DUMP_TRACE_DIR=. \
    gst-launch-1.0 audiotestsrc num-buffers=1000 ! vorbisenc ! vorbisdec ! fakesink
$ ls -al *.gsttrace
$ gst-report-1.0  pipeline0.gsttrace
ELEMENT        %CPU   %TIME   TIME
vorbisenc0      59.8   86.6    200 ms
vorbisdec0       8.2   11.9   27.5 ms
fakesink0        1.1    1.5   3.58 ms
pipeline0        0.0    0.0      0 ns
audiotestsrc0    0.0    0.0      0 ns
```

##### How to generate an instrumentation graph
```bash
$ gst-report-1.0  --dot pipeline0.gsttrace | dot -Tpng > perf.png
$ eog ./perf.png 
```
<img src=gst-instruments-perf.png border=0></img>


#### Case study: Unit test in NNStreamer

```bash
nnstreamer/test$ ./testAll.sh 1
nnstreamer/test$ cd performance/profile/tensor_convertor
$ eog ${number_of_test_case}.svg
```

Then, you can see a time cost and a CPU usage in pipeline.


### HawkTracer
HawkTracer is a highly portable, low-overhead, configurable profiling tool built in Amazon Video for getting performance metrics from low-end devices.
The library provides many different types of events (e.g. CPU usage event, duration tracepoint), but the list can easily be extended by the user.
* https://amzn.github.io/hawktracer/index.html (Doxygen book for HawkTracer)

#### Features
* Low CPU/memory overhead
* Multi-platform
* Highly configurable on runtime
* Easy build integration (CMake & pkg-config files)
* Pre-defined and user-defined events
* Simple macros for code instrumentation
* Streaming events to file, through TCP/IP protocol, or handling them in custom function
* Client for receiving event stream
* Library for parsing event stream (so you can write your own client)

#### How to build a library
```bash
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .   # Instead of make
```

#### Attaching HawkTracer for profiling

* CMake-based projects
If you use CMake build system, you can use following code to attach HawkTracer library to your project:
```bash
project(your_project)

# optionally, you might define a path to HawkTracer's CMake module
# CMake the path below should be a path to a directory where HawkTracerConfig.cmake is located, e.g.:
# list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/HawkTracer")

find_package(HawkTracer REQUIRED)

add_executable(your_project main.cc)

target_link_libraries(your_project HawkTracer::hawktracer)
```
* pkg-config
HawkTracer library provides pkg-config file which can be used to find required libraries and include paths. You can simply integrate it e.g. with your compilation command:
```bash
$ g++ my_project.cpp $(pkg-config --cflags --libs hawktracer)
```

#### Initialize  HawkTracer library
There are 2 functions which always have to be called in projects profiled by HawkTracer: ht_init and ht_deinit. Additionally, you need to specify an event listener. HawkTracer currently provides 2 listeners:
* TCP listener, which streams events over the network
* File dump listener, which saves events to a file.
```bash
int main(int argc, char** argv)
{
  ht_init(argc, argv); // initialize library
  HT_Timeline* timeline = ht_global_timeline_get(); // timeline, where all events are posted. You can define your own timeline, or use global timeline
  HT_FileDumpListener* listener = ht_file_dump_listener_create("file_name.htdump", buffer_size, NULL); // initialize listener
  const size_t buffer_size = 4096; // size of internal listener's buffer
  ht_timeline_register_listener(timeline, ht_file_dump_listener_callback, listener); // register listener to a timeline

  // your code goes here...
  
  ht_timeline_flush(timeline); // flush all remaining events from timeline
  ht_timeline_unregister_all_listeners(timeline); // unregister listeners from timeline
  ht_file_dump_listener_destroy(listener); // deinitialize listener
  ht_deinit(); // uninitialize library

  return 0;
}
```

#### Instrumenting source code

The library provides a few helper macros for reporting data to a timeline:
```bash
// Pushes any type of event to a timeline
HT_TIMELINE_PUSH_EVENT(TIMELINE, EVENT_TYPE, EVENT_PARAMETERS,...)

// Reports a duration of specific block of code (available only for C++ or C GNU compiler)
HT_TP_STRACEPOINT(TIMELINE, LABEL)

// The same as above, but automatically sets label to current function name
HT_TP_FUNCTION(TIMELINE)
```

There are few macros which report events to a global timeline, they're prefixed with G_:
```bash
HT_TP_G_STRACEPOINT(LABEL)
HT_TP_G_FUNCTION()
```

#### Collect the data
The code registers file dump listener, which saves all the events to a file file_name.htdump.
Assuming your events have been saved to file_name.htdump file, you can generate the JSON file running following command:
```bash
$ hawktracer-to-json --source file_name.htdump --output output_file.json
```

#### Analyzing the data
* First of all, install google-chrome (or chromium-brwser) in you own PC. 
* Open **chrome://tracing/ webapge after running the browser.
* Click load buttong. Then, open  file output_file.json
* You should see a callstack with timing

<img src=hawktracer-chrome-tracing-out.png border=0></img>
