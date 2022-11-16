---
title: macOS
...

# Installing NNStreamer on macOS

**DISCLAIMER**: This manual is work-in-progress and needs to be updated.
Specifically, only PyTorch support is covered for now.

## Prerequisites

The following dependencies are needed for installing and building:

* Apple's Clang >= 10.0.0
* [Homebrew](https://brew.sh/) for installing dependencies

## Installing via Homebrew

The simplest way to install NNStreamer on macOS is through our third-party 
Homebrew repository:

```bash
$ brew tap nnstreamer/neural-network
$ brew install nnstreamer
```

## Building from source

Note that most frameworks will be disabled during the configuration, if you 
manage to make NNStreamer work on this platform with your preferred framework,
please provide a pull request with an update of this doc!

Install the necessary dependencies:

```bash
$ brew install meson ninja pkg-config cmake libffi glib \
    gstreamer gst-plugins-base gst-plugins-good numpy
```

If you need to run PyTorch models:

```bash
$ brew install libtorch
```

Clone NNStreamer's repository:

```bash
$ git clone https://github.com/nnstreamer/nnstreamer.git
$ cd nnstreamer
```

Optionally checkout a recent version more stable than the main branch, for 
instance:

```bash
$ git checkout v2.1.1
```

Configure the build with [meson](https://mesonbuild.com):

```bash
$ meson build \
    --prefix=/usr/local \
    -Dwerror=false -Denable-test=false
```

The target files will be installed in `/usr/local/lib/gstreamer-1.0`, 
`/usr/local/lib/nnstreamer`, `/usr/local/include/nnstreamer` and 
`/usr/local/etc/`.

Finally build and install using [ninja](https://ninja-build.org/):

```bash
$ ninja -C build install
```


