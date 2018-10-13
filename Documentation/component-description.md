# Gstreamer Stream Data Types

- other/tensor
- other/tensors
- other/tensorsave (Planned)

findtype specifics: refer to the wiki (other/tensorsave)

Note that in any frame in a stream of other/tensor and other/tensors, there should be only ONE instance of other/tensor or other/tensors.

Note that except for tensor\_decoder, which accepts data semantics from pipeline developers, and tensor\_converter, which accepts data semantics from sink pad caps, any NNStreamer tensor plugins or any instances of tensor streams should be agnostic to the data semantics. With data semantics, we know whether the corresponding tensor denotes for video, audio, text, or any other "meaningful data types". The NNStreamer elements and data-pipeline should treat every other/tensor and other/tensors as arrays of numbers only.

# Gstreamer Elements (Plugins)

Note that "stable" does not mean that it is complete. It means that it has enough test cases and complies with the overall design; thus, "stable" features probably won't be modified extensively. Features marked "experimental" can be modified extensively due to its incomplete design and implementation or crudeness.

In this page, we focus on the status of each elements. For requirements and designs of each element, please refer to the README.md of the element.

- [tensor\_converter](../gst/tensor_converter/)
  - Video (stable)
    - video/x-raw. Colorspaces of RGB, BGRx, Gray8 are supported.
    - Caution: if width is not divisible by 4, RGB/Gray8 video incurs memcpy.
  - Audio (stable)
    - audio/x-raw. Users should specify the number of frames in a single buffer, which denotes the number of frames in a single tensor frame with the property of ```frames-per-buffer```. Needs more test cases.
  - Text (stable)
    - text/x-raw. Partially implemented. Needs fixes and test cases.
  - Binary (experimenal)
    - application/octet-stream. Stream pipeline developer MUST specify the corresponding type and dimensions via properties (input-dim, input-type)
- [tensor\_filter](../gst/tensor_filter/)
  - Main (stable)
    - Supported features
      - Fixed input/ouput dimensions (fixed by subplugin)
      - Flexible dimensions (output dimension determined by subplugin according to the input dimension determined by pipeline initialization)
      - Invoke subplugin with pre-allocated buffers
      - Invoke subplugin and let subplugin allocate output buffers.
      - Accept other/tensors.
    - TODO: Allow to manage synchronization policies.
    - TODO: Recurrent models. The first target unit-test is LSTM.
  - Tensorflow-Lite (stable)
  - Custom (stable)
  - Tensorflow (WIP. Not working properly as of 2018/10/13)
  - Other NNFW TBD (caffe2, caffe, ...)
- [tensor\_sink](../gst/tensor_sink/) (stable)
- [tensor\_transform](../gst/tensor_transform/) (stable, but with a lot of WIP items)
  - Supported features
    - Type Cast (typecast) (stable)
    - Dimension Change (dimchg) (experimental)
    - Arithmetic (arithmetic) (experimental. stable with limited sub features)
    - Transpose (transpose) (experimental. stable with limited sub features)
    - Standardization/Normalization (stand) (experimental. stable with limited sub features)
    - More features coming soon!
- [tensor\_merge](../gst/tensor_merge/) (stable, but with NYI WIP items)
- [tensor\_split](../gst/tensor_split/) (stable, but with NYI WIP items)
- [tensor\_decoder](../gst/tensor_decoder/) (stable, but with NYI WIP items)
  - Supported features
    - Direct video conversion (video/x-raw) (stable)
    - Image classification labeling (text/x-raw) (stable)
    - Bounding Boxes (video/x-raw) (WIP)
    - More items are planned.
- [tensor\_mux](../gst/tensor_mux/) (stable)
- [tensor\_demux](../gst/tensor_demux/) (stable)
- [tensor\_source](../gst/tensor_source/) (planned)
- [tensor\_save](../gst/tensor_save/) (planned)
- [tensor\_load](../gst/tensor_load/) (planned)
- [tensor\_aggregator](../gst/tensor_aggregator) (stable)
- [tensor\_ros\_sink](../gst/tensor_ros_sink/) (planned)
- [tensor\_ros\_src](../gst/tensor_ros_src/) (planned)


Note that test elements in /tests/ are not elements for applications. They exist as scaffoldings to test the above elements especially in the case where related elements are not yet implemented.

# Other Components
- CI ([@private server](http://nnsuite.mooo.com/) (stable): Up and running
- CI (@AWS) (experimental): WIP (sewon-oh, leemgs)
- CD (@launchpad / @OBS) (planned)
- [Test cases](tests/): Mandatory unit test cases required to pass for each PR.
  - Planned to use [SSAT](https://github.com/myungjoo/SSAT) later.
  - Each element and feature is required to register its testcases here.
- [Example applications](nnstreamer_example/): Example gstreamer applications using nnstreamer and example sub-plugins for nnstreamer. The binaries from this directory is not supposed to be packaged with the main binary package.
- Packaing for Distros / SW-Platform Compatibility.
  - [Tizen](packaging/) (stable): RPM packaging for Tizen 5.0+. It is expected to be compatible with other RPM-based distros; however, it is not tested or guaranteed.
  - [Ubuntu](debian/) (stable): DEB packaging for Ubuntu 16.04. It is highly expected to be compatible with later versions as well; but, not tested yet. Debian is not tested, either.
  - Yocto (experimental)
  - Android (planned with high priority)
  - iOS (planned with low priority)
- [Common library](common/)
- [Common headers](include/)
