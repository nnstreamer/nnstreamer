---
title: Component description
...

# Gstreamer Stream Data Types

- other/tensor (obsolete! use ```other/tensors,num_tensors=1``` instead)
- other/tensors

Each frame of an ```other/tensor``` or ```other/tensors``` stream should have only ONE instance of other/tensor or other/tensors.

Except ```tensor_decoder```, which accepts data semantics from pipeline developers, and ```tensor_converter```, which accepts data semantics from sink pad caps, every NNStreamer tensor plugin and tensor stream should be agnostic to the data semantics. With data semantics, we know whether the corresponding tensor denotes for video, audio, text, or any other "meaningful data types". The NNStreamer elements and data-pipeline, being agnostic to such semantics, should treat every other/tensor and other/tensors as multi-dimensional arrays of general numbers. This allows inter-operability between different neural network models and frameworks. The semantics of tensors should be handled and encapsulated by their direct accessors: the pipeline writer (or the application) and the corresponding framework (and its subplugin).

# Gstreamer Elements (Plugins)

Note that "stable" does not mean that it is complete. It means that it has enough test cases and complies with the overall design; thus, "stable" features probably won't be modified extensively. Features marked "experimental" can be modified extensively due to its incomplete design and implementation or crudeness. "Planned" is still in the works so it will be released soon.

In this page, we focus on the status of each elements. For requirements and designs of each element, please refer to the README.md of the element. Refer to the papers, [ICSE2021](https://arxiv.org/pdf/2101.06371) and [ICSE2022](https://arxiv.org/abs/2201.06026) for alternative descriptions on these elements along with some figures.

- [tensor\_converter](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_converter)
  - Video (stable)
    - '''video/x-raw'''. Colorspaces of RGB, BGRx, Gray8 are supported.
    - Caution: if width is not divisible by 4, RGB/Gray8 video incurs memcpy.
  - Audio (stable)
    - '''audio/x-raw'''. Users should specify the number of frames in a single buffer, which denotes the number of frames in a single tensor frame with the property of ```frames-per-buffer```. Needs more test cases.
  - Text (stable)
    - '''text/x-raw'''. Users should specify the byte size of a single tensor frame with the property ```input-dim```. Needs more test cases.
  - Binary (stable)
    - '''application/octet-stream'''. Stream pipeline developer MUST specify the corresponding type and dimensions via properties (input-dim, input-type)
  - Custom subplugins (stable)
    - FlatBuf (stable): '''other/flatbuf-tensor''' --> '''other/tensors'''
    - ProtoBuf (stable): '''other/protobuf-tensor''' --> '''other/tensors'''
    - FlatBuf::FlatBuf (stable): '''other/flexbuf''' --> '''other/tensors'''
    - Python3 (stable): You may define your own conversion mechanism with python script.
    - Developers may add their own custom converter subplugin with the APIs defined in ```nnstreamer_plugin_api_converter.h```. Such subplugins may be added in run-time, which is supposed to be installed at the path designated by ```decoders``` path in ```nnstreamer.ini```.
- [tensor\_filter](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_filter)
  - Main (stable)
    - Supported features
      - Fixed input/output dimensions (fixed by subplugin)
      - Flexible dimensions (output dimension determined by subplugin according to the input dimension determined by pipeline initialization)
      - Invoke subplugin with pre-allocated buffers
      - Invoke subplugin and let subplugin allocate output buffers.
      - Accept ```other/tensors``` with static and flexible format.
      - Users can add subplugins in run-time.
    - TODO: Allow to manage synchronization policies.
  - Custom-C (stable)
  - Custom-C++-Class (stable)
  - Custom-C-Easy (stable) (single function ops. basis for lambda functions in the future)
  - Custom-Python (stable) (~~2.7 and~~ 3)
  - Custom-LUA (stable)
  - Custom Native Functions (stable) (Supply custom-filter in run-time)
  - Tensorflow (stable) (1.09, 1.13, 2.3, 2.7, 2.8 tested)
  - Tensorflow-lite (stable) (1.09, 1.13, 2.3, 2.7, 2.8 tested)
      - For tensorflow-lite version 2.x, use ```tensorflow2-lite``` as the subplugin name, which allows to use both tensorflow-lite 1.x and 2.x simultaneously in a pipeline.
  - Caffe2 (stable)
  - PyTorch (stable)
  - TVM (stable)
  - NNTrainer (stable. **maintained by its own community**)
  - TRIx NPU/Samsung (stable. **maintained by the manufacturer**)
  - Movidius-X NCS2/Intel (stable)
  - NNFW-Runtime/nnfw (stable)
  - Edge-TPU/Google (stable)
  - openVINO/Intel (stable)
  - ARMNN (stable)
  - SNPE/Qualcomm (stable)
  - Vivante/Verisilicon (stable)
  - TensorRT/NVidia (stable)
  - SNAP (stable)
  - Deepview-RT/NXP (stable. **maintained by the manufacturer**)
  - MXNet (experimental)
  - Mediapipe (experimental)
  - WIP: NCNN
  - [Guide on writing a filter subplugin](writing-subplugin-tensor-filter.md)
  - [Codegen and code template for tensor\_filter subplugin](https://github.com/nnstreamer/nnstreamer-example/tree/main/templates)
- [tensor\_transform](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_transform) (stable)
  - Supported features
    - Type Cast (typecast) (stable, orc supported with the property ```acceleration```)
    - Dimension Change (dimchg) (stable with limited sub features)
    - Arithmetic (arithmetic) (stable, orc supported with the property ```acceleration```)
    - Transpose (transpose) (stable with limited sub features)
    - Standardization/Normalization (stand) (stable with limited sub features)
    - More features coming soon!
- [tensor\_decoder](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_decoder) (stable, but with NYI WIP items)
  - Supported features
    - Direct video conversion (video/x-raw) (stable)
    - Image classification labeling (text/x-raw) (stable)
    - Bounding boxes (video/x-raw) (stable)
        - This supports different standards, which can be configured at run-time.
    - Image segmentation (video/x-raw) (stable) and depth
    - Body pose (video/x-raw) (stable)
    - Flatbuf (stable)
    - Flexbuf (stable)
    - Protobuf (stable)
    - binary/octet-stream (stable)
  - Users can add plugins in run-time.
- [tensor\_sink](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_sink) (stable)
  - ```appsink```-like element, which is specialized for ```other/tensors```. You may use appsink with capsfilter instead.
- [tensor\_merge](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_merge) (stable)
  - This combines muiltiple single-tensored (```other/tensors,num_tensors=1```) streams into a single-tensored stream by merging dimensions of incoming tensor streams. For example, it may merge two ```dimensions=640:480``` streams into ```dimensons=1280:480```, ```dimensions=640:960```, or ```dimensions=640:480:2```, according to a given configuration.
  - Users can adjust sync-mode and sync-option to change its behaviors of when to create output tensors and how to choose input tensors.
  - Users can adjust how dimensions are merged (the rank merged, the order of merged streams).
- [tensor\_split](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_split) (stable)
  - This is the opposite of ```tensor_merge```. This splits a single-tensored (```other/tensors,num_tensors=1```) stream into multiple single-tensored streams. For example, a stream of ```dimensions=1920:1080``` may split into ```dimensions=1080:1080``` and ```dimensions=840:1080```.
  - Users can adjust how dimensions are split
- [tensor\_mux](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_mux) (stable)
  - This combines multiple ```other/tensor(s)``` streams into a single ```other/tensors``` stream while keeping the input stream dimensions. Thus, the number of tensors (```num_tensors```) increase accordingly without changing dimensions of incoming tensors. For example, merging the two tensor streams, ```num_tensors=1,dimensions=3:2``` and ```num_tensors=1,dimensions=4:4:4``` becomes ```num_tensors=2,dimensions=3:2,4:4:4```, combining frames from the two streams, enforcing synchronization.
  - Both merge and mux combine multiple streams into a stream; however, merge combines multiple tensors into a tensor, updating the dimensions while mux keep the tensors and combine them into a single container.
  - Users can adjust sync-mode and sync-option to change its behaviors of when to create output tensors and how to choose input tensors..
- [tensor\_demux](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_demux) (stable)
  - This decomposes multi-tensor (```num_tensors > 1```) tensor streams into multiple tensor streams without touching their dimensions. For example, we may split a tensor stream of ```num_tensors=3,dimensions=5,4,3``` into ```num_tensors=2,dimensions=5,4``` and ```num_tensors=1,dimensions=3```. Users may configure how the tensors split into (e.g., from ```num_tensors=6```, into 3:2:1, 4:2, 1:1:1:1:1:1, or so on, reordering as well).
- [tensor\_aggregator](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_aggregator) (stable)
  - This combines multiple frames of tensors into a frame in a single tensor stream. For example, it may aggregate two frames into a frame and reduce the framerate into half: ```dimensions=300:300,framerate=30/1``` --> ```dimensions=300:300:2,framerate=15/1```.
  - Users can adjust how frames are aggregated including how many frames are aggregated, how many frames are skipped after each aggregation, which frames are aggregated, which dimension is merged, and so on.
- [tensor\_repo\_sink](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_repo) (stable)
  - This allows to create circular tensor streams by pairing up with ```tensor_repo_src```. Although gstreamer does not allow circular streams, with a pair of ```tensor_repo_sink/src``` we can transmit tensor data without actually connecting gstreamer src/sink pads. It is called ```tensor_repo_*``` because the src/sink pair shares a tensor repository.
  - In the pair, ```tensor_repo_sink``` is the entering point of the tensor frames. When you create a circular stream, sending back tensors from "behind" to the "front", this element is supposed to be located at the "behind".
- [tensor\_repo\_src](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_repo) (stable)
  - This allows to create circular tensor streams by pairing up with ```tensor_repo_sink```. Although gstreamer does not allow circular streams, with a pair of ```tensor_repo_sink/src``` we can transmit tensor data without actually connecting gstreamer src/sink pads. It is called ```tensor_repo_*``` because the src/sink pair shares a tensor repository.
  - In the pair, ```tensor_repo_src``` is the exit point of the tensor frames. When you create a circular stream, sending back tensors from "behind" to the "front", this element is supposed to be located at the "front".
- [tensor\_if](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_if) (stable)
  - This element controls the flow or tensor data based on the given decision condition and the input tensor data. Unlike other similar gstreamer elements, including ```valve```, ```input-selector```, or ```output-selector```, which decides based on the property value given by threads out of the pipeline, this element, ```tensor_if```, decides based on the stream data in the pipeline. Thus, pipelines can switch between their sub-pipelines (e.g., input nodes, output nodes, and processing nodes) precisely (without losing a frame or two) if they should decide based on an inference result or sensor data.
  - This element allows a lot of varying configurations and users can even provide a C function callback for conditions; please refer to its documentation.
- [tensor\_sparse\_enc](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_sparse) (stable)
  - This transforms ```other/tensors,format=static``` to ```other/tensors,format=sparse```, encoding tensor data frames that may compress data size of sparse tensors.
- [tensor\_sparse\_dec](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_sparse) (stable)
  - This transforms ```other/tensors,format=sparse``` to ```other/tensors,format=static```.
- [tensor\_query\_client](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_query) (stable)
  - This element sends queries to and receives answers from ```tensor_query_server{sink, src}``` elements. This works as if this is a ```tensor_filter``` with a remote processing element. This is a basic among-device AI capability that is supposed to offload inference workloads to different devices.
- [tensor\_query\_serversrc](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_query) (stable)
  - This element receives queries from remote (out of its pipeline) ```tensor_query_client``` or its compatible component of ```nnstreamer-edge```.
  - This element behaves as an entry point of a server or service element for remote clients, accepting offload requests.
  - Users constructing a "server" pipeline are supposed to use this element as an entry point (input node).
  - If you use service construct of ML-Service-API, you need a single pair of ```tensor_query_server{src, sink}``` in your registered pipeline.
- [tensor\_query\_serversink](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_query) (stable)
  - This element sends back answers of given queries to remote (out of its pipeline) ```tensor_query_client```, which is connected to the paired ```tensor_query_serversrc```. The server elements are supposed to be paired-up so that the query-sending client gets the corresponding answers.
  - Users constructing a "server" pipeline are supposed to use this element as an exit point (output node).
- [tensor\_crop](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_crop) (stable)
  - This element crops a tensor stream based on the values of another tensor stream. Unlike the conventional gstreamer crop elements, which crop data frames based on the property values given outside from the pipeline, this element crop data frames based on the streamed values in the pipeline. Thus, users can crop tensors with the inference results or sensor data directly without involving external threads; e.g., cropping out detected objects from a video stream, to create a video stream focussing on a specific object. This element uses flexible tensors because the crop-size varies dynamically.
- [tensor\_rate](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_rate) (stable)
  - This element controls a frame rate of tensors streams. Users can also control QoS with throttle property.
- [tensor\_src\_iio](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer/tensor_source) (stable)
  - Requires GStreamer 1.8 or above.
  - Creates tensor streams from Linux iio (sensors) device nodes.
- [tensor\_src\_tizensensor](https://github.com/nnstreamer/nnstreamer/tree/main/ext/nnstreamer/tensor_source) (stable)
  - This element imports data from Tizen sensor framework, which can provide sensor-fusion service, and generates tensor stream from it. Obviously, this works only in Tizen.
- [tensor\_src\_grpc](https://github.com/nnstreamer/nnstreamer/tree/main/ext/nnstreamer/tensor_source) (stable)
  - This element generates tensor streams from data received via grpc connection.
- [tensor\_sink\_grpc](https://github.com/nnstreamer/nnstreamer/tree/main/ext/nnstreamer/tensor_sink) (stable)
  - This element sends data via grpc connection from tensor streams.
- [android supports](https://github.com/nnstreamer/nnstreamer/tree/main/ext/nnstreamer/android_source) (stable)
  - This element allows to accept data streams from Android's media framework.


Elements that do not deal with ```other/tensors``` streams, but are in this repo:

- [join](https://github.com/nnstreamer/nnstreamer/tree/main/gst/join) (stable)
  - This element combines multiple streams into a stream. This does not merge dimensions or data frames. This simply forwards every incoming frames from multiple sources into a single destination, combining the data stream paths only.
- [mqttsrc](https://github.com/nnstreamer/nnstreamer/tree/main/gst/mqtt) (stable)
  - This element receives tensor stream data via MQTT protocol.
  - With "mqtt-hybrid" mode, data streams (TCP/direct) can be separated from control streams (MQTT) to increase data throughput.
- [mqttsink](https://github.com/nnstreamer/nnstreamer/tree/main/gst/mqtt) (stable)
  - This element sends tensor stream data via MQTT protocol.
  - With "mqtt-hybrid" mode, data streams (TCP/direct) can be separated from control streams (MQTT) to increase data throughput.


Elements dealing with ```other/tensors``` streams, but are in a different repo:
- [tensor\_ros\_sink](https://github.com/nnstreamer/nnstreamer-ros) (stable for ROS1/ROS2)
  - You may send tensor streams via ROS pub/sub structure.
- [tensor\_ros\_src](https://github.com/nnstreamer/nnstreamer-ros) (stable for ROS1/ROS2)
  - You may receive tensor streams via ROS pub/sub structure.

Note that test elements in /tests/ are not elements for applications. They exist as scaffoldings to test the above elements especially in the case where related elements are not yet implemented.



# Other Components
- CI ([@AWS](http://ci.nnstreamer.ai/nnstreamer/ci/taos)) (stable): Up and Running.
- CD
  - Tizen (since 5.5 M1) [Package Download](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) [Build & Release Infra](https://build.tizen.org/project/show/Tizen:Unified)
  - Ubuntu [Launchpad PPA](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa)
  - Yocto/OpenEmbedded [OpenEmbedded Layer, "meta-neural-network"](https://layers.openembedded.org/layerindex/branch/master/layer/meta-neural-network/)
  - Android WIP: JCenter Repository & Daily Build Release
  - macOS WIP: Daily Build Release
- [Test cases](https://github.com/nnstreamer/nnstreamer/tree/main/tests/): Mandatory unit test cases required to pass for each PR.
  - Used [SSAT](https://github.com/myungjoo/SSAT).
  - Each element and feature is required to register its testcases at [test case directory](https://github.com/nnstreamer/nnstreamer/tree/main/tests/)
- Examples: Example GStreamer applications using NNStreamer and example sub-plugins for NNStreamer. The binaries from this directory is not supposed to be packaged with the main binary package.
  - [Example GStreamer applications](https://github.com/nnstreamer/nnstreamer-example)
  - [Example sub-plugins](https://github.com/nnstreamer/nnstreamer/tree/main/nnstreamer_example)
- Packaging for Distros / SW-Platform Compatibility.
  - [Tizen](https://github.com/nnstreamer/nnstreamer/tree/main/packaging) (stable): RPM packaging for Tizen 5.0+. It is expected to be compatible with other RPM-based distros; however, it is not tested or guaranteed.
  - [Ubuntu](https://github.com/nnstreamer/nnstreamer/tree/main/debian) (stable): DEB packaging for Ubuntu 16.04. It is highly expected to be compatible with later versions as well; but, not tested yet. Debian is not tested, either.
  - [Yocto](https://github.com/nnsuite/meta-nerual-network) (stable)
  - [Android](https://github.com/nnstreamer/nnstreamer/tree/main/jni) (stable)
  - macOS (built & tested w/ macOS. but packaging is not provided, yet.)
  - iOS (planned with low priority)
- [Common headers](https://github.com/nnstreamer/nnstreamer/tree/main/gst/nnstreamer)
- [Change Log](https://github.com/nnstreamer/nnstreamer/tree/main/CHANGES)
