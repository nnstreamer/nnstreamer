# Gstreamer Stream Data Types

- other/tensor
- other/tensors
- other/tensorsave (Planned)

findtype specifics: refer to the wiki ([other/tensorsave](https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind))

Note that in any frame in a stream of other/tensor and other/tensors, there should be only ONE instance of other/tensor or other/tensors.

Note that except for tensor\_decoder, which accepts data semantics from pipeline developers, and tensor\_converter, which accepts data semantics from sink pad caps, any NNStreamer tensor plugins or any instances of tensor streams should be agnostic to the data semantics. With data semantics, we know whether the corresponding tensor denotes for video, audio, text, or any other "meaningful data types". The NNStreamer elements and data-pipeline should treat every other/tensor and other/tensors as arrays of numbers only.

# Gstreamer Elements (Plugins)

Note that "stable" does not mean that it is complete. It means that it has enough test cases and complies with the overall design; thus, "stable" features probably won't be modified extensively. Features marked "experimental" can be modified extensively due to its incomplete design and implementation or crudeness.

In this page, we focus on the status of each elements. For requirements and designs of each element, please refer to the README.md of the element.

- [tensor\_converter](../gst/nnstreamer/tensor_converter)
  - Video (stable)
    - video/x-raw. Colorspaces of RGB, BGRx, Gray8 are supported.
    - Caution: if width is not divisible by 4, RGB/Gray8 video incurs memcpy.
  - Audio (stable)
    - audio/x-raw. Users should specify the number of frames in a single buffer, which denotes the number of frames in a single tensor frame with the property of ```frames-per-buffer```. Needs more test cases.
  - Text (stable)
    - text/x-raw. Partially implemented. Needs fixes and test cases.
  - Binary (stable)
    - application/octet-stream. Stream pipeline developer MUST specify the corresponding type and dimensions via properties (input-dim, input-type)
- [tensor\_filter](../gst/nnstreamer/tensor_filter)
  - Main (stable)
    - Supported features
      - Fixed input/ouput dimensions (fixed by subplugin)
      - Flexible dimensions (output dimension determined by subplugin according to the input dimension determined by pipeline initialization)
      - Invoke subplugin with pre-allocated buffers
      - Invoke subplugin and let subplugin allocate output buffers.
      - Accept other/tensors.
    - TODO: Allow to manage synchronization policies.
  - Custom-C (stable)
  - Custom-Python (stable)
  - Custom Native Functions (stable) (Supply custom-filter in run-time)
  - Tensorflow (stable)
  - Tensorflow-lite (stable)
  - Caffe2 (stable)
  - PyTorch (stable)
  - Movidius-X NCS2 (stable)
  - Other NNFW TBD (keras, caffe, Exynos-NPU, edge-TPU, Qualcomm-SNPE, ...)
  - [Guide on writing a filter subplugin](writing-subplugin-tensor-filter.md)
- [tensor\_sink](../gst/nnstreamer/tensor_sink) (stable)
- [tensor\_transform](../gst/nnstreamer/tensor_transform) (stable)
  - Supported features
    - Type Cast (typecast) (stable, orc supported with the property ```acceleration```)
    - Dimension Change (dimchg) (stable with limited sub features)
    - Arithmetic (arithmetic) (stable, orc supported with the property ```acceleration```)
    - Transpose (transpose) (stable with limited sub features)
    - Standardization/Normalization (stand) (stable with limited sub features)
    - More features coming soon!
- [tensor\_merge](../gst/nnstreamer/tensor_merge) (stable)
- [tensor\_split](../gst/nnstreamer/tensor_split) (stable)
- [tensor\_decoder](../gst/nnstreamer/tensor_decoder) (stable, but with NYI WIP items)
  - Supported features
    - Direct video conversion (video/x-raw) (stable)
    - Image classification labeling (text/x-raw) (stable)
    - Bounding Boxes (video/x-raw) (stable)
    - More items are planned.
- [tensor\_mux](../gst/nnstreamer/tensor_mux) (stable)
- [tensor\_demux](../gst/nnstreamer/tensor_demux) (stable)
- [tensor\_source](../gst/nnstreamer/tensor_source) (stable for IIO. More sources coming soon)
- [tensor\_save](../gst/nnstreamer/tensor_saveload) (planned)
- [tensor\_load](../gst/nnstreamer/tensor_saveload) (planned)
- [tensor\_aggregator](../gst/nnstreamer/tensor_aggregator) (stable)
- [tensor\_ros\_sink](https://github.com/nnsuite/nnstreamer-ros) (stable for ROS1)
- [tensor\_ros\_src](https://github.com/nnsuite/nnstreamer-ros) (stable for ROS1)
- [tensor\_repo\_sink](../gst/nnstreamer/tensor_repo) (stable)
- [tensor\_repo\_src](../gst/nnstreamer/tensor_repo) (stable)


Note that test elements in /tests/ are not elements for applications. They exist as scaffoldings to test the above elements especially in the case where related elements are not yet implemented.

# API Support

- C-API
  - Main target is Tizen, but supports other OS as well.
  - [Implementation](../api/capi) (stable, since Tizen 5.5 M2)
- C#-API (.NET)
  - Main target is Tizen, but supports other OS as well.
  - [Implementation](https://github.com/Samsung/TizenFX/tree/master/src/Tizen.MachineLearning.Inference) (to be released for Tizen 5.5 M2)
- JAVA-API (Android)
  - [Android sample app](https://github.com/nnsuite/nnstreamer-example/tree/master/android/example_app/api-sample) uses JAVA APIs to implement Android-NNStreamer apps.
  - We will soon deploy NNStreamer via JCenter: org.nnsuite.nnstreamer
  - Note that the Android Sample Applications published via Google Play Store, [Source Code](https://github.com/nnsuite/nnstreamer-example/tree/master/android/example_app), are developed before NNStreamer Java API. They use GStreamer functions.

# Other Components
- CI ([@AWS](http://nnsuite.mooo.com/nnstreamer/ci/taos)) (stable): Up and Running.
- CD
  - Tizen (since 5.5 M1) [Package Download](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) [Build & Release Infra](https://build.tizen.org/project/show/Tizen:Unified)
  - Ubuntu [Launchpad PPA](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa)
  - Yocto/OpenEmbedded [OpenEmbedded Layer, "meta-neural-network"](https://layers.openembedded.org/layerindex/branch/master/layer/meta-neural-network/)
  - Android WIP: JCenter Repository & Daily Build Release
  - macOS WIP: Daily Build Release
- [Test cases](../tests/): Mandatory unit test cases required to pass for each PR.
  - Used [SSAT](https://github.com/nnsuite/SSAT).
  - Each element and feature is required to register its testcases at [test case directory](../tests/)
- Examples: Example GStreamer applications using NNStreamer and example sub-plugins for NNStreamer. The binaries from this directory is not supposed to be packaged with the main binary package.
  - [Example GStreamer applications](https://github.com/nnsuite/nnstreamer-example)
  - [Example sub-plugins](../nnstreamer_example)
- Packaing for Distros / SW-Platform Compatibility.
  - [Tizen](../packaging) (stable): RPM packaging for Tizen 5.0+. It is expected to be compatible with other RPM-based distros; however, it is not tested or guaranteed.
  - [Ubuntu](../debian) (stable): DEB packaging for Ubuntu 16.04. It is highly expected to be compatible with later versions as well; but, not tested yet. Debian is not tested, either.
  - [Yocto](https://github.com/nnsuite/meta-nerual-network) (stable)
  - [Android](../jni) (stable)
  - macOS (built & tested w/ macOS. but packaging is not provided, yet.)
  - iOS (planned with low priority)
- [Common headers](../gst/nnstreamer)
- [Change Log](../CHANGES)
