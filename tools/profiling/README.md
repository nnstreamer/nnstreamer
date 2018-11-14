
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
