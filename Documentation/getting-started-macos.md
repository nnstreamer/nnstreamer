# Using NNStreamer in macOS

## DISCLAIMER

This manual is work-in-progress and will be progressively updated. In detail, 1) how to run the tests and examples provided with NNStreamer, 2) how to enable and use [TensorFlow](https://www.tensorflow.org/) filters, and 3) how to enable custom filters written by python will be updated very soon. Other things such as how to enable full features of NNStreamer on macOS will be also added later.

## Table of contents

* [Prerequisites](#Prerequisites)
* Easy way: [Install via Brew taps (third-party repository)](#Install-via-Brew-taps-(third-party-repository))
* Starting from scratch: [Build from source](#Build-from-source)

## Prerequisites

The following dependencies are needed to compile/build/run.

* clang/clang++ >= 10.0.0

```bash
$ xcode-select --install
```

* [Homebrew](https://brew.sh/)

```bash
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew install cask
$ brew update
```

## Install via Brew taps (third-party repository)

This is the most simple way to install NNStreamer into your macOS system.

```bash
$ brew tap nnsuite/neural-network
$ brew install nnstreamer
```

## Build from source

Install a bunch of packages required to build and run NNStreamer using brew.

```bash
$ brew install meson ninja pkg-config cmake libffi glib gstreamer gst-plugins-base gst-plugins-good numpy
```

Clone the master (default) branch of the GitHub repository of NNStreamer.

```bash
$ git clone https://github.com/nnsuite/nnstreamer.git
Cloning into 'nnstreamer'...
...omission...
Checking out files: 100% (326/326), done.

$ ls
nnstreamer

$ cd nnstreamer
```

(Optional) You can choose the latest stable release of NNStreamer instead of the bleeding-edge version.

```bash
$ git tag
v0.0.1
v0.0.2
v0.0.3
v0.1.0
v0.1.1
v0.1.2
v0.2.0
v0.3.0

$ git checkout v0.3.0
Note: checking out 'v0.3.0'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by performing another checkout.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -b with the checkout command again. Example:

  git checkout -b <new-branch-name>

HEAD is now at f60e840 Release of 0.3.0
```

Configure using [meson](https://mesonbuild.com).

```bash
$ PKG_CONFIG_PATH=/usr/local/opt/libffi/lib/pkgconfig \
meson build \
--prefix=/usr/local \
-Denable-tensorflow=false \
-Denable-tensorflow-lite=false \
-Denable-pytorch=false -Denable-caffe2=false

The Meson build system
Version: 0.51.1
Source dir: /Users/wooksong/Work/tmp/nnstreamer
Build dir: /Users/wooksong/Work/tmp/nnstreamer/build
Build type: native build
Project name: nnstreamer
Project version: 0.3.0
C compiler for the build machine: cc (clang 10.0.0 "Apple LLVM version 10.0.0 (clang-1000.10.44.4)")
C++ compiler for the build machine: c++ (clang 10.0.0 "Apple LLVM version 10.0.0 (clang-1000.10.44.4)")
C compiler for the host machine: cc (clang 10.0.0 "Apple LLVM version 10.0.0 (clang-1000.10.44.4)")
C++ compiler for the host machine: c++ (clang 10.0.0 "Apple LLVM version 10.0.0 (clang-1000.10.44.4)")
Build machine cpu family: x86_64
Build machine cpu: x86_64
Compiler for C supports arguments -Wredundant-decls: YES
Compiler for C++ supports arguments -Wredundant-decls: YES
Compiler for C supports arguments -Wwrite-strings: YES
Compiler for C++ supports arguments -Wwrite-strings: YES
Compiler for C supports arguments -Wformat: YES
Compiler for C++ supports arguments -Wformat: YES
Compiler for C supports arguments -Wformat-nonliteral: YES
Compiler for C++ supports arguments -Wformat-nonliteral: YES
Compiler for C supports arguments -Wformat-security: YES
Compiler for C++ supports arguments -Wformat-security: YES
Compiler for C supports arguments -Winit-self: YES
Compiler for C++ supports arguments -Winit-self: YES
Compiler for C supports arguments -Waddress: YES
Compiler for C++ supports arguments -Waddress: YES
Compiler for C supports arguments -Wno-multichar: YES
Compiler for C++ supports arguments -Wno-multichar: YES
Compiler for C supports arguments -Wvla: YES
Compiler for C++ supports arguments -Wvla: YES
Compiler for C supports arguments -Wpointer-arith: YES
Compiler for C++ supports arguments -Wpointer-arith: YES
Compiler for C supports arguments -Wmissing-declarations: YES
Compiler for C supports arguments -Wmissing-prototypes: YES
Compiler for C supports arguments -Wnested-externs: YES
Compiler for C supports arguments -Waggregate-return: YES
Compiler for C supports arguments -Wold-style-definition: YES
Compiler for C supports arguments -Wdeclaration-after-statement: YES
Found pkg-config: /usr/local/bin/pkg-config (0.29.2)
Run-time dependency glib-2.0 found: YES 2.60.7
Run-time dependency gobject-2.0 found: YES 2.60.7
Run-time dependency gstreamer-1.0 found: YES 1.16.0
Run-time dependency gstreamer-base-1.0 found: YES 1.16.0
Run-time dependency gstreamer-controller-1.0 found: YES 1.16.0
Run-time dependency gstreamer-video-1.0 found: YES 1.16.0
Run-time dependency gstreamer-audio-1.0 found: YES 1.16.0
Run-time dependency gstreamer-app-1.0 found: YES 1.16.0
Run-time dependency gstreamer-check-1.0 found: YES 1.16.0
Library m found: YES
Library dl found: YES
Run-time dependency threads found: YES
Found CMake: /usr/local/bin/cmake (3.15.3)
Run-time dependency protobuf found: NO (tried pkgconfig, cmake and framework)
Run-time dependency orc-0.4 found: YES 0.4.29
Program orcc found: YES (/usr/local/bin/orcc)
Program pkg-config found: YES (/usr/local/bin/pkg-config)
Run-time dependency python-2.7 found: YES 2.7
Has header "numpy/arrayobject.h" : NO
Run-time dependency python3 found: YES 3.7
Has header "numpy/arrayobject.h" : NO
Run-time dependency opencv found: NO (tried pkgconfig, cmake and framework)
Run-time dependency libpng found: YES 1.6.37
Program cp found: YES (/bin/cp)
Run-time dependency GTest found: YES 1.9.0
Configuring nnstreamer-test.ini using configuration
Configuring nnstreamer.ini using configuration
Configuring nnstreamer.pc using configuration
Build targets in project: 27
Found ninja-1.9.0 at /usr/local/bin/ninja
```

Build and install using [ninja](https://ninja-build.org/). In this instruction, the target files will be installed into the directories, /usr/local/lib/gstreamer-1.0 /usr/local/lib/nnstreamer, /usr/local/include/nnstreamer, and /usr/local/etc. You can change the installation root directory (i.e., /usr/local) by providing the other directory for the --prefix option of meson.

```bash
$ ninja -C build install

ninja: Entering directory `build'
[94/95] Installing files.
Installing gst/nnstreamer/libnnstreamer.dylib to /usr/local/lib/gstreamer-1.0
Installing gst/nnstreamer/libnnstreamer.a to /usr/local/lib
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_direct_video.dylib to /usr/local/lib/nnstreamer/decoders
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_direct_video.a to /usr/local/lib
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_image_labeling.dylib to /usr/local/lib/nnstreamer/decoders
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_image_labeling.a to /usr/local/lib
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_bounding_boxes.dylib to /usr/local/lib/nnstreamer/decoders
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_bounding_boxes.a to /usr/local/lib
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_pose_estimation.dylib to /usr/local/lib/nnstreamer/decoders
Installing ext/nnstreamer/tensor_decoder/libnnstreamer_decoder_pose_estimation.a to /usr/local/lib
Installing /Users/wooksong/Work/tmp/nnstreamer/gst/nnstreamer/tensor_typedef.h to /usr/local/include/nnstreamer
Installing /Users/wooksong/Work/tmp/nnstreamer/gst/nnstreamer/tensor_filter_custom.h to /usr/local/include/nnstreamer
Installing /Users/wooksong/Work/tmp/nnstreamer/gst/nnstreamer/nnstreamer_plugin_api_filter.h to /usr/local/include/nnstreamer
Installing /Users/wooksong/Work/tmp/nnstreamer/gst/nnstreamer/nnstreamer_plugin_api_decoder.h to /usr/local/include/nnstreamer
Installing /Users/wooksong/Work/tmp/nnstreamer/gst/nnstreamer/nnstreamer_plugin_api.h to /usr/local/include/nnstreamer
Installing /Users/wooksong/Work/tmp/nnstreamer/build/nnstreamer.ini to /usr/local/etc
Installing /Users/wooksong/Work/tmp/nnstreamer/build/nnstreamer.pc to /usr/local/lib/pkgconfig
```


