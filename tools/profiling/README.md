
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
