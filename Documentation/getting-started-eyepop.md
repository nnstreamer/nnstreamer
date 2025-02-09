---
title: EyePop
...

# Installing NNStreamer for EyePop development

These instructions supplement the docs for installing the nnstreamer package for Ubuntu:
* [With an existing .deb package](./getting-started-ubuntu-ppa.md)
* [By building a .deb package locally](./getting-started-ubuntu-debuild.md)
* [By building through meson/ninja](./getting-started-meson-build.md)

See those docs for more details about `nnstreamer` internals.

Currently supported EyePop platforms:
* Ubuntu 22.04 on `amd64` (fully supported)
* Ubuntu 22.04 on `aarch64` (TensorFlow Lite only)

## Development Environment Setup

For preexisting Docker images with all dependencies installed, see the
[Docker setup instructions][docker-setup]. The corresponding Dockerfiles can be found
[here][dockerfiles].

To develop on your host machine, install the following required packages through your system package
manager (instructions provided for `apt`). For EyePop-specific dependencies, add the EyePop package
repo to your sources list before proceeding:

```sh
sudo echo "deb [trusted=yes] http://repo.dev.eyepop.xyz.s3.us-east-1.amazonaws.com/ stable main" \
    | sudo tee /etc/apt/sources.list.d/eyepop.dev.list > /dev/null
```

`apt` instructions for installing everything below (leave out `eyepop-torchvision` on `aarch64`):

```sh
sudo apt install build-essential debhelper devscripts gcc9 meson ninja-build libglib2.0-dev \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libjson-glib-dev gstreamer1.0-tools \
     gstreamer1.0-plugins-good libgtest-dev libopencv-dev flex bison python3-dev libpaho-mqtt-dev \
     eyepop-onnxruntime eyepop-tflite eyepop-torchvision
```

Build tools:
* `bison`
* `build-essential`
* `debhelper`
* `devscripts`
* `gcc-9`
* `flex`
* `meson`
* `ninja-build`
* `python3-dev`

Build dependencies:
* `libglib2.0-dev`
* `libgstreamer1.0-dev`
* `libgstreamer-plugins-base1.0-dev`
* `libgtest-dev`
* `libjson-glib-dev`
* `libopencv-dev`
* `libpaho-mqtt-dev`

Runtime dependencies:
* `gstreamer1.0-tools`
* `gstreamer1.0-plugins-good`

EyePop-specific dependencies:
* `eyepop-onnxruntime`
* `eyepop-tflite`
* `eyepop-torchvision` (amd64 only)

## Build/install instructions (Linux)

### Developing locally

`nnstreamer` uses `meson` to orchestrate the build. Use individual feature flags to control what you
build for a particular platform.

To build/test for Ubuntu 22.04 on `amd64` with support for ONNX, TensorFlow, and TensorFlow Lite:

```sh
meson -Dwerror=false -Donnxruntime-support=enabled -Dtf-support=enabled -Dcaffe2-support=disabled -Dpython3-support=disabled build/ -Denable-test=false

ninja -C build

ninja -C build test
```

To build/test for Ubuntu 22.04 on `aarch64` with support for TensorFlow Lite and ONNX only:

```sh
meson -Dwerror=false -Donnxruntime-support=enabled -Dtf-support=disabled -Dcaffe2-support=disabled -Dpython3-support=disabled build/ -Denable-test=false

ninja -C build

ninja -C build test
```

### Building installable packages locally

To build installable `.deb` packages locally, use `debuild`. Like `eyepop-ml-dev`, the EyePop build
of `nnstreamer` uses platform identifiers to customize how packages get built.

| Platform                       | CUDA Support? | Identifier                 |
|--------------------------------|---------------|----------------------------|
| Linux, `amd64`, Ubuntu 22.04   | yes           | `linux-amd64-jammy-cuda`   |
| Linux, `amd64`, Ubuntu 22.04   | no            | `linux-amd64-jammy`        |
| Linux, `aarch64`, Ubuntu 22.04 | no            | `linux-aarch64-jammy`      |

To build for Ubuntu 22.04 on `amd64` with CUDA:

```sh
DEB_BUILD_OPTIONS="nocheck notest" debuild -b -us -uc -Plinux-amd64-jammy-cuda
```

To build for Ubuntu 22.04 on `amd64` without CUDA:

```sh
DEB_BUILD_OPTIONS="nocheck notest" debuild -b -us -uc -Plinux-amd64-jammy
```

To build for Ubuntu 22.04 on `aarch64`:

```sh
DEB_BUILD_OPTIONS="nocheck notest" debuild -b -us -uc -Plinux-aarch64-jammy
```

This will generate a series of `.deb` packages in the repo's parent directory. To install them with
`apt`, run the following:

```sh
sudo apt install ../nnstreamer*.deb
```

## Build/install instructions (Windows on ARM)

### Initial setup

Set up a Windows on ARM development environment as specified in the `eyepop-edge-sdk-windows` docs
including the extraction of the ONNX DLLs.

Install build dependencies through `pacman`:

```sh
pacman -S \
  mingw-w64-clang-aarch64-clang \
  mingw-w64-clang-aarch64-cmake \
  mingw-w64-clang-aarch64-gtest \
  mingw-w64-clang-aarch64-make \
  mingw-w64-clang-aarch64-pkgconf \
  mingw-w64-clang-aarch64-python3.12
```

Install build dependencies through `pip` in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip3 install meson ninja
```

Install the runtime dependencies through `pacman`

```sh
pacman -S \
  mingw-w64-clang-aarch64-glib2 \
  mingw-w64-clang-aarch64-libffi
```

### Build and install

The following assumes that a `$WORKSPACE` environment variable has been set and that
`$WORKSPACE/install_dev` points to an install directory containing necessary dependencies (i.e. 
ONNX).

Starting from the `eyepop-nnstreamer` directory:

```sh
meson setup builddir \
        --prefix=$WORKSPACE/install_dev \
        -Donnxruntime-support=enabled -Dlua-support=disabled \
        -Dwerror=false -Denable-test=false -Denable-nnstreamer-check=false \
        -Dpkg_config_path=$WORKSPACE/install_dev/lib/pkgconfig:/usr/lib/pkgconfig \
    && meson install -C builddir
```

### Notes on Windows-specific fixes

The upstream `nnstreamer` project does not support Windows, so care has been taken to split out any
Windows-specific modifications.

In general, the following strategies have been used to keep Windows code separate:
* In Meson files, Windows changes are guarded by `if build_platform == 'windows'`
* In C/C++ source, Windows changes are guarded by `#if (defined(_WIN32) || defined(__CYGWIN__))`

### Next steps

* Figure out a better way to detect if we're building in a Windows environment in the top-level
  `meson.build`
* Consider replacing `NNS_API_IMPORTS` with `NNS_API_EXPORTS` and setting that on all libraries (see
  `nnstreamer.api.h`)
* Fix orc generation rather than relying on the cached `windows/nnstreamer-orc.h`/`.cpp`
* Add proper error handling for `char*` to `wchar_t*` conversions
* Make `nnstreamer` tests work

## Notes on packaging strategy

See `eyepop-ml-dev`'s README.

[docker-setup]: https://app.gitbook.com/o/lp5TZAZIwu5jXdzZth9T/s/0fWYYHCrIOcShgRMiWhf/readme/howtos/running-as-docker-container
[dockerfiles]: https://github.com/eyepop-ai/eyepop-docker-images
