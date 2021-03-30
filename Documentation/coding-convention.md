---
title: Coding Convention
...

# Coding Convention

In general, NNStreamer Application follows [The Coding Style of GStreamer](https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/developing.html#what-is-the-coding-style-for-gstreamer-code) for coding convention.

## C codes (.c sources)

All C codes of NNStreamer are required to use K&R with 2-space indenting. Just follow what's already there and you'll be fine. We only require .c files to be indented, headers may be indented differently for better readability. Please use spaces for indenting, not tabs, even in header files.

When you push your commits, apply `gst-indent` to .c files before you submit a commit with style change.
If there is a change due to code style issues, make two separate commits: (Please do not include other codes' style change in the same commit)
- commit with style change only (i.e., commit gst-indent-formatted original code - not your code change)
- commit with your code change only (i.e., contents only).

```
$ ./tools/development/gst-indent <file-name>
```

## C headers (.h)

You may indent differently from what gst-indent does. You may also break the 80-column rule with header files.

Except the two, you are required to follow the general coding styles mandated by gst-indent

## C++ files (.cc)

Do not use .cpp extensions, use .cc extensions for C++ sources. Use .h for headers.

We do not have strict rules on C++ sources, yet, because all the cores are written in C.
However, please try to stick with the same indentation rules (2 spaces) and common senses.

## Other files

- [Java] TBD
- [Python] TBD
- [Bash] TBD


# File Locations

## Directory structure of nnstreamer.git

- **api**: API definitions and implementations
    - **android**: Android APIs
    - **capi**: C-APIs (Tizen and others)
    - Note that .NET APIs are located at https://github.com/Samsung/TizenFX along with other Tizen .NET APIs
- **debian**: Debian/Ubuntu packaging files
- **Documentation**: Documentations
- **ext/nnstreamer**: NNStreamer plugins and subplugins that depend on optional or non-standard external packages. Components in this directory can be built and included optionally.
    - **android\_source**: Plugins required to support Android/JAVA APIs.
    - **tensor\_decoder**: Decoder subplugins
    - **tensor\_filter**: Filter subplugins
    - **tensor\_source**: Source elements that provide tensor/tensors streams. We do not have subplugin architectures for sources.
    - Potentially we can add **tensor\_converter** here later for converter subplugins. We do not have converter subplugins released, yet.
- **gst/nnstreamer**: All core nnstreamer codes are located here.
    - **tensor\_\* **: Plugins of nnstreamer.
- **jni**: Android/Java build scripts.
- **nnstreamer\_example**: Example custom filters required by test cases. In the old days, we used to have all nnstreamer examples here. Most of such examples are moved to [Example git](https://github.com/nnstreamer/nnstreamer-example) except for those who required by test cases.
- **packaging**: Tizen RPM build scripts. OpenSUSE/Redhat Linux may reuse this.
- **tests**: Unit test cases. We have SSAT test cases and GTEST test cases. There are a lot of subdirectories, which are groups of unit test cases.
- **tools**: Various developmental tools and scripts of NNStreamer.

## Related git repositories

- [NNStreamer Example Applications \& Documents](https://github.com/nnstreamer/nnstreamer-example)
- [TAOS-CI, CI Service for On-Device AI Systems](https://github.com/nnstreamer/TAOS-CI)
- [NNStreamer ROS (Robot OS) Support](https://github.com/nnstreamer/nnstreamer-ros)
- [NNStreamer Android Build Resource](https://github.com/nnstreamer/nnstreamer-android-resource): additional files required by Android builds.
- [ORC, aarch64 support](https://github.com/nnsuite/orc): we are going to support aarch64-ORC to accelerate transform operations of nnstreamer in aarch64 devices.
- [NNStreamer-Edge](https://github.com/nnstreamer/nnstreamer-edge): WIP
- [NNStreamer Yocto/OpenEmbedded Layer](https://github.com/nnstreamer/meta-neural-network): refer to [Openembedded layer page](https://layers.openembedded.org/layerindex/branch/master/layer/meta-neural-network/)
- [NNStreamer Web Page](https://github.com/nnstreamer/nnstreamer.github.io): WIP
- **tizenport-\* **: Tizen-ROS support. Refer to [build.tizen.org](https://build.tizen.org/project/show/devel:AIC:Tizen:5.0:nnsuite)



## Related external git repositories

- [NNStreamer mirror in Tizen.org](https://git.tizen.org/cgit/platform/upstream/nnstreamer/)
