
## Tracing

### Using GstShark
[GstShark](https://developer.ridgerun.com/wiki/index.php?title=GstShark) is an open-source project from Ridgerun that provides benchmarks and profiling tools for GStreamer 1.7.1 (and above).
It includes tracers for generating debug information plus some tools to analyze the debug information.
GstShark provides easy to use and useful tracers, paired with analysis tools to enable straightforward optimizations.
GstShark leverages GStreamer's tracing hooks and open-source and standard tracing and plotting tools to simplify the process of understanding the bottlenecks in your pipeline.

```bash
$ sudo apt install libgstreamer1.0-dev
$ sudo apt install graphviz libgraphviz-dev
$ sudo apt install octave epstool babeltrace
$ git clone https://github.com/RidgeRun/gst-shark/
$ cd gst-shark
$ ./autogen.sh  --disable-gtk-doc  --prefix /usr/ --libdir /usr/lib/x86_64-linux-gnu/
$ make
$ sudo make install
```

#### Tracers of GstShark
* InterLatency: Measures the latency time at different points in the pipeline.
* ProcTime: Measures the time an element takes to produce an output given the corresponding input.
* Framerate:    Measures the amount of frames that go through a src pad every second.
* ScheduleTime: Measures the amount of time between two consecutive buffers in a sink pad. T
* CPUUsage: Measures the CPU usage every second. In multiprocessor systems this measurements are presented per core.
* Graphic:  Records a graphical representation of the current pipeline.
* Bitrate:  Measures the current stream bitrate in bits per second.
* Queue Level:  Measures the amount of data queued in every queue element in the pipeline.
* Buffer:   Prints information of every buffer that passes through every sink pad in the pipeline. This information contains PTS and DTS, duration, size, flags and even refcount.

#### Generating  trace files
The user has the capability of selecting the tracers that will be run with the pipeline by listing them (separated by the character ";") using the option "GST_TRACER_PLUGINS" or "GST_TRACERS", depending on the GStreamer version, at the time of running the pipeline in a very similar way than the "GST_DEBUG" option.
```bash
$ export GST_SHARK_LOCATION=./
# For GStreamer 1.7.1
$ GST_DEBUG="GST_TRACER:7" GST_TRACER_PLUGINS="cpuusage;proctime;framerate"\
     gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink
# For GStreamer 1.8.1 and later
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="cpuusage;proctime;framerate"\
     gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink
```

#### Case study

* Example: Processing time
The GstShark processing time tracer ("proctime") provides information to the user about the amount of time that each element of the pipeline is taking for processing each data buffer that goes through it. In other words, it measures the time every element needs to process a buffer, allowing to know which element takes too much time completing its tasks, causing a slow performance, among others issues.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="proctime" gst-launch-1.0 videotestsrc num-buffers=60 ! \
tee name=tee0 tee0. ! queue ! identity sleep-time=10000 ! fakesink tee0. ! queue ! \
identity sleep-time=30000 ! fakesink tee0. ! queue ! identity sleep-time=50000 ! fakesink
$ ../scripts/graphics/gstshark-plot ./gstshark_2018-11-06_20\:19\:58/ -s proctime.pdf
```

