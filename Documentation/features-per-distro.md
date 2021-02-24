## Feature lists of each official release

This document lists features enabled in official releases of each OS or Linux distro.

Note that users may adjust features freely with build options if they decide to build nnstreamer; any Linux developers should be able to do it with ease.

This document reflects the status of 2021-02-24 ebafe31 (1.7.1 devel)


## Tizen

- Main GStreamer Elements
    - All main tensor-\* elements are enabled including IIO support.
    - ROS extensions are **not** released. With ROS packages, you may build and deploy.
- Extra Elements and Subplugins
    - Enabled
        - tensorflow-lite (1.13)
        - tensorflow2-lite (2.3) w/ gpu delegation
        - nnfw runtime (a.k.a. ONERT)
        - python filter
        - mvncsdk2
        - openvino
        - edgeTPU
        - flatbuf (converter/decoder)
        - protobuf (converter/decoder)
        - Tizen sensor source
        - all trivial filter/converter/decoder subplugins.
    - Disabled
        - tensorflow (we do not use Bazel in Tizen)
        - tensorflow2
        - armnn (ready, but no one requested)
        - verisilicon/vivante (ready, a few vivante libraries are not ported to Tizen public distro)
        - pytorch (ready)
        - caffe2 (ready)
        - snap (not supported in Tizen)
        - snpe (not supported in Tizen)


## Android

Android releases will have multiple binaries with different configurations; e.g., "nnstreamer-full", "nnstreamer-lite", "nnstreamer-basic", and so on.

This list shows a release corresponding to "full".

- GStreamer Elements
    - All main tensor-\* elements except for IIO
    - ROS extensions are **not** released.
- Extra Elements and Subplugins
    - Enabled
        - tensorflow-lite (1.13)
        - tensorflow2-lite (2.3) w/ gpu delegation
        - nnfw runtime (a.k.a. ONERT)
        - (depending on options) snap / snpe (you cannot install both simultaneously)
        - all trivial filter/converter/decoder subplugins.
    - Disabled
        - tensorflow
        - tensorflow2
        - armnn
        - verisilicon/vivante
        - pytorch
        - caffe2
        - python filter
        - mvncsdk2
        - openvino
        - edgeTPU
        - and others.

## Ubuntu

We limit the number of extra subplugins in Ubuntu PPA. However, users can easily build more subplugins easily with meson build options or they may build their own subplugins independently with the given header files (nnstreamer-dev).

- GStreamer Elements
    - All main tensor-\* elements.
    - ROS extensions are **not** released. Users may build theirs with given ROS packages.
- Extra Elements and Subplugins
    - Enabled
        - tensorflow
        - tensorflow-lite
        - protobuf
        - flatbuf
        - caffe2
        - edgetpu
        - openvino
        - python
        - pytorch
    - Disabled
        - verisilicon/vivante
        - armnn
        - tensorflow2
        - tensorflow2-lite
        - nnfw runtime
        - snap
        - snpe
        - and others
