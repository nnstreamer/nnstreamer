
## Tracing

### Using GstShark
[GstShark](https://developer.ridgerun.com/wiki/index.php?title=GstShark) is an open-source project from Ridgerun that provides benchmarks and profiling tools for GStreamer 1.7.1 (and above).
It includes tracers for generating debug information plus some tools to analyze the debug information.
GstShark provides easy to use and useful tracers, paired with analysis tools to enable straightforward optimizations.
GstShark leverages GStreamer's tracing hooks and open-source and standard tracing and plotting tools to simplify the process of understanding the bottlenecks in your pipeline.

Currently, GstShark is not stable. So, if you find a new issue in your own development environment, Please write an issue at the below github.com webpage.
For technical questions, please send an email to support@ridgerun.com.

* https://developer.ridgerun.com/wiki/index.php?title=GstShark
* https://github.com/RidgeRun/gst-shark/

#### How to compile the GstShark code
In this section, We assume that you use Ubuntu linux distribution. This guide is written on Ubuntu 16.04 distribution.
```bash
$ sudo apt install libgstreamer1.0-dev
$ sudo apt install graphviz libgraphviz-dev
$ sudo apt install octave epstool babeltrace
$ git clone https://github.com/RidgeRun/gst-shark/
$ cd gst-shark
$ git checkout -b v0.5.3 v0.5.3
$ ./autogen.sh  --disable-gtk-doc  --prefix /usr/ --libdir /usr/lib/x86_64-linux-gnu/
$ make -j`nproc`
$ sudo make install
```

#### Tracers of GstShark
* ProcTime: Measures the time an element takes to produce an output given the corresponding input.
* InterLatency: Measures the latency time at different points in the pipeline.
* Framerate:    Measures the amount of frames that go through a src pad every second.
* ScheduleTime: Measures the amount of time between two consecutive buffers in a sink pad. T
* CPUUsage: Measures the CPU usage every second. In multiprocessor systems this measurements are presented per core.
* Graphic:  Records a graphical representation of the current pipeline.
* Bitrate:  Measures the current stream bitrate in bits per second.
* Queue Level:  Measures the amount of data queued in every queue element in the pipeline.
* Buffer:   Prints information of every buffer that passes through every sink pad in the pipeline.

#### Case study

##### Getting started: How to generate trace files
The user has the capability of selecting the tracers that will be run with the pipeline by listing them (separated by the character ";") using the option "GST_TRACER_PLUGINS" or "GST_TRACERS", depending on the GStreamer version, at the time of running the pipeline in a very similar way than the "GST_DEBUG" option.
```bash
$ unset GST_SHARK_LOCATION
$ gst-launch-1.0 --version
# For GStreamer 1.7.1
$ GST_DEBUG="GST_TRACER:7" GST_TRACER_PLUGINS="cpuusage;proctime;framerate"\
     gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink
# For GStreamer 1.8.1 and later
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="cpuusage;proctime;framerate"\
     gst-launch-1.0 videotestsrc ! videorate max-rate=15 ! fakesink
```


##### Processing time tracer
The GstShark processing time tracer ("proctime") provides information to the user about the amount of time that each element of the pipeline is taking for processing each data buffer that goes through it. In other words, it measures the time every element needs to process a buffer, allowing to know which element takes too much time completing its tasks, causing a slow performance, among others issues.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="proctime" gst-launch-1.0 videotestsrc num-buffers=60 ! \
tee name=tee0 tee0. ! queue ! identity sleep-time=10000 ! fakesink tee0. ! queue ! \
identity sleep-time=30000 ! fakesink tee0. ! queue ! identity sleep-time=50000 ! fakesink
```


##### Interlatency tracer
The interlatency tracer measures the time that a buffer takes to travel from one point to another inside the pipeline. The total latency of the pipeline is the time that the buffer needs to go from the source to the sink element (most downstream). However, the interlatency tracer makes it possible to know what route of the pipeline is adding more time to the overall latency by displaying the exact time that the buffer took to go through every element on the pipeline.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="interlatency" gst-launch-1.0 videotestsrc ! queue ! \
videorate max-rate=15 ! fakesink sync=true
```


##### Framerate tracer
The frame rate of a pipeline is one of the most useful characteristics when debugging, especially for those which include live video source elements. It is common that the frame rate is a design parameter and a requirement for declaring the performance of a pipeline as successful. Also, the frame rate is usually used for determining if the output of pipeline is synced, when both video and audio are involved. The frame rate is the measurement of the frame frequency, that means that it is the measurement of the number of frames that go through the source pad of certain element in a given time. Normally frame rate is expressed in frames per second (FPS).

The framerate tracer displays the number of frames that go through every source pad of every element of the pipeline and it is updated and printed on the output log every second.
```bash
$ cd ./scripts/graphics
GST_DEBUG="GST_TRACER:7" GST_TRACERS="framerate" gst-launch-1.0 videotestsrc ! \
videorate max-rate=15 ! fakesink sync=true
```


##### Schedule time tracer
The schedule time tracer is very similar to the processing time tracer. The processing time is focused on measuring the time that an element takes to completely process a given buffer and that is why it is measured on the source pad of that element, because when the buffer gets to that point is when the element has finished its process and the buffer is ready to be pushed to next element. On the other side, what the scheduling time tracer does is measuring the elapsed time since one buffer arrived to the sink pad of the element with the immediate consecutive one, without taking in consideration the time needed for the element to compute the arrived data buffer.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="scheduletime" gst-launch-1.0 \
videotestsrc is-live=true do-timestamp=true num-buffers=50 ! \
'video/x-raw, format=(string)YUY2, width=(int)1280, height=(int)720, framerate=(fraction)10/1' ! \
videoconvert ! queue ! avenc_h263p ! fakesink sync=true
```

##### CPU usage tracer
The CPU usage tracer measures the load on the CPU at the time of running a determined pipeline. This gives the user a way of knowing immediately if the host system is capable of running the pipeline without overcharging the cores of the CPU. It is important to mention that the measurements displayed every second by the tracer correspond to the total load of the CPU and it is not the exact data of the load caused by the pipeline at the time of running it. However, the measurements made and printed on the output log give a good idea of how the load of the CPU is behaving at the time of running the pipeline. With this information, it is possible to check if the pipeline produces an unexpected and undesired increase of the load of the CPU that could cause a failure on the system or a failure of the pipeline due to lack of resources.

The CPU usage tracer has the capability of measuring every core of the system. In a multiprocessor system, the output log will include the load of each individual core, giving an idea of the effect that the pipeline has on the total CPU consumption on the system.

Currently this tracer is only available for GNU/Linux based systems, since the method used to determine the load on every core available is by reading the /proc/stat file. 

```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="cpuusage" gst-launch-1.0 videotestsrc ! \
'video/x-raw, format=(string)YUY2, width=(int)640, height=(int)480, framerate=(fraction)30/1' ! \
 videorate max-rate=30 ! videoconvert ! queue ! avenc_h263p ! queue ! avimux ! fakesink sync=true
```


##### Graphic tracer
The graphic tracer is an add-on of the GstShark tracer suit. It does not generate any output log providing any information about the performance of the pipeline neither it helps the user with the debugging process at the time of facing an unexpected issue on the behavior of a pipeline. However, this tracer is very useful since it shows to the user the pipeline **graphically** so the user can easily inspect it visually.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="graphic;cpuusage;framerate" gst-launch-1.0 videotestsrc num-buffers=500 ! \
'video/x-raw, format=(string)YUY2, width=(int)640, height=(int)480, framerate=(fraction)30/1' ! \
videorate max-rate=30 ! videoconvert ! queue ! avenc_h263p ! queue ! avimux ! fakesink sync=true
$ ls -al ./graphic/
$ xdot  ./graphic/pipeline.dot
```
<img src=gstshark_graphic_tracer.png border=0></img>

##### Bitrate tracer
This tracer is similar to the framerate tracer, but instead of measuring the number of frames produced every second, it measures bits.

The bitrate tracer displays the number of bits that go out of a source pad in a second. This is a measurement of the bitrate of the stream. It is updated and printed on the output log every second. The data from this tracer is specially useful at the output of encoders, as it will be able to provide the compression rate and the potential variation of the bitrate value over time.

Currently, there are no plots available to be generated based on this tracer. This is a feature that is being developed and will be made available in a future release.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="bitrate" gst-launch-1.0 videotestsrc ! \
'video/x-raw, format=(string)YUY2, width=(int)640, height=(int)480, framerate=(fraction)30/1' ! \
videorate max-rate=30 ! videoconvert ! queue ! avenc_h263p ! queue ! avimux ! fakesink sync=true
```


##### Queue level tracer
The Queue Level tracer measures the amount of data queued in every queue element in the pipeline. A new trace will be printed every time a buffer enters a queue.
This data is specially useful when debugging latency and performance bottlenecks. The queue level will provide the developer with additional data to debug queue underruns and other related problems.
Currently, there are no plots available to be generated based on this tracer. This is a feature that is being developed and will be made available in a future release.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="queuelevel" gst-launch-1.0 videotestsrc ! \
'video/x-raw, format=(string)YUY2, width=(int)640, height=(int)480, framerate=(fraction)30/1' ! \
videorate max-rate=30 ! videoconvert ! queue max-size-buffers=20 ! avenc_h263p ! \
queue max-size-time=400000000 ! avimux ! fakesink sync=true
```


##### Buffer tracer
The Buffer tracer prints out information about each buffer that goes out of a source pad of an element. Each output log line includes several fields with information about each buffer that goes out of a source pad of an element. Data includes pts, dts, duration, offset, offset_end, size, flags and refcount.
```bash
$ cd ./scripts/graphics
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="buffer" gst-launch-1.0 videotestsrc ! fakesink sync=true
```

#### gstshark-plot (Experimental/Unstable)
gstshark-plot is a set of [Octave](https://www.gnu.org/software/octave/) scripts included with GstShark. The gstshark-plot scripts are located in scripts/graphics directory, inside the repository. The main script that processes the data is the gstshark-plot script. Currently, the scripts need to be run on this directory, but on upcoming releases the scripts will be accessible from any path. Make sure the GST_SHARK_CTF_DISABLE environment variable is unset, to enable the generation of the full traces.
Note that you have to run "unset GST_SHARK_LOCATION" statement in order to archive output date into CTF (Commen Trace Format, ./gstshark_yyyy-mm-dd_hh:mm:ss/) folder.
* CTF (Common Trace Format) file: Directory with date and time with the traces of the latest session. 
```bash
$ unset GST_SHARK_LOCATION
$ unset GST_SHARK_CTF_DISABLE
$ cd ./gst-shark/scripts/graphics/
$ GST_DEBUG="GST_TRACER:7" GST_TRACERS="proctime" gst-launch-1.0 videotestsrc num-buffers=60 ! \
tee name=tee0 tee0. ! queue ! identity sleep-time=10000 ! fakesink tee0. ! queue ! \
identity sleep-time=30000 ! fakesink tee0. ! queue ! identity sleep-time=50000 ! fakesink
$
$ $ tree  ./gstshark_2018-11-20_18\:37\:54/
./gstshark_2018-11-20_18:37:54/
|-- datastream
|-- graphic
|   `-- pipeline.dot
`-- metadata
1 directory, 3 files
$ ./gstshark-polot --help
$ ./gstshark-plot {path-of-CTF-output-folder} -s trace.{pdf|png}
Then, The GNU plot graph will be automatically appeared on a pop-up window.
```

As an alternative method, you can also try to use the experiental Eclipse plug-in at the below webpage.
* https://developer.ridgerun.com/wiki/index.php?title=GstShark_-_Install_Eclipse_plugin
