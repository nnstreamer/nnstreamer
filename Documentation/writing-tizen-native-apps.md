# Writing a Tizen Native App

This document guides how to write a Tizen Native (C/C++) app with nnstreamer APIs (Tizen Machine-Learning Inference API Set).

## About the Tizen ML-Inference API Sets

There are two API sets as of Tizen 5.5 M2, Machine-Learning Inference: "Pipeline" and "Single".

With **Pipeline** API set, you can construct a GStreamer pipeline with nnstreamer elements and gstreamer elements in the "whitelist". GStreamer plugins not listed in the "whitelist" is not supported by Pipelie APIs. Note that an alias element, ```tizencamvideosrc``` (or ML_TIZEN_CAM_VIDEO_SRC macro), denotes a video camera source, which may vary per device; in other words, you don't need to worry whether it's ```v4l2src``` or ```camsrc``` or whatsoever. For the list of whitelist elements, refer to the Tizen API doc or ```/etc/nnstreamer.ini```.

With **Single** API set, you can invoke a neural network model with a input tensor/tensors without constructing a full pipeline.

## General Flow / Single

1. Get a handle (open a model)

2. (Optional) Investigate the required input/output types (dimensions) with utility functions.

3. Invoke (provide an input, get an output)

4. Close the handle

Note that set_timeout API allows you to set timeout for invoke function.

## General Flow / Pipeline

It allows the flexibility to consturct a stream pipeline with complex topology including multiple neural networks and frameworks, different pre/post-processors, data and path manipulators, and various input and output nodes. This is far more rich featured compared to Single API; however, the developer is required to understand GStreamer pipelines. For GStreamer pipelines in general, please refer to GStreamer documents.

1. Construct a pipeline (ml\_pipeline\_construct) and get a pipeline handle.

2. (Optional) Attach callbacks or get other handles from the constructed pipeline. In order to attach callbacks or get handles, the corresponding elements should have names defined so that they can be recognized in run-time.

3. Start the pipeline. The callbacks will be invoked during the pipeline execution. You may control it with the additional handles (e.g., input/output switches, valves).

4. Stop/Restart the pipeline.

5. Close the pipeline handle.

### Sink Callbacks (ml\_pipeline\_sink\_cb)

This allows you to provide a function (callback), which is invoked whenever an output tensor/tensors is available. This can be attached to ```tensor_sink``` and ```appsink``` with the given name.

### State Callbacks (ml\_pipeline\_state\_cb)

This allows you to provide a function (callback), which is invoked whenever the states of the constructed pipeline change.

### Source Handle

1. Get src\_handle from a pipeline with a name of the appsrc element.
2. Push input data with the src\_handle.
3. Release the src\_handle.

### Switch/Valve

1. Get switch/valve handle.
2. Control the handle.
3. Release the handle.

### Utility Functions

You may investigate the data types/dimensions/names or get performance-related data.


### Element Whitelist

The elements that can be included in the pipeline (with construct API) are limited by the whitelist (defined by ```/etc/nnstreamer.ini```).

However, with an internal API, allowed to platform binaries only (.rpm. not .tpk), the whitelist policy doesn't apply. The package, ```nnstreamer-tizen-internal-capi-devel```, provides internal APIs; use it with BuildRequires from your platform package .spec file.


# Tizen Sample Native Apps

## Single

A simple sample Tizen app with Single APIs is at (nnstreamer-example.git, "SingleSample")[https://github.com/nnsuite/nnstreamer-example/tree/master/Tizen.native/SingleSample]

## Pipeline

A simple sample Tizen app with Pipeline APIs is at (nnstreamer-example.git, "Application")[https://github.com/nnsuite/nnstreamer-example/tree/master/Tizen.native/Application]


# Tizen ML API Documentation

The official Tizen ML API Documentation will be available after the release of Tizen 5.5. Before the release you may create doxygen documents created from /api/capi/include and /api/capi/doc


