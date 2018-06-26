# NNStreamer

Neural Network Streamer for AI Projects.
* http://suprem.sec.samsung.net/confluence/display/STAR/NN+Streamer

## Objectives

- Provide neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
  - **Efficient Streaming for AI Projects**: Neural network models wanted to use efficient and flexible streaming management as well. 
  - **Intelligent Media Filters!**: Use a neural network model as a media filter / converter.
  - **Composite Models!**: Allow to use multiple neural network models in a single stream instance.
  - **Multi Modal Intelligence!**: Allow to use multiple sources for neural network models.

## Maintainers
* MyungJoo Ham (myungjoo.ham@samsung.com)

## Committers
* Jijoong Mon (jijoon.moon@samsung.com)
* Geunsik Lim (geunsik.lim@samsung.com)
* Sangjung Woo (sangjung.woo@samsung.com)
* Wook Song (wook16.song@samsung.com)
* Jaeyun Jung (jy1210.jung@samsung.com)
* Jinhyuck Park (jinhyuck83.park@samsung.com)
* Hyoungjoo Ahn (hello.ahn@samsung.com)
* Sewon Oh (sewon.oh@samsung.com)

## Components

Note that this project has just started and most of the components are in design phase.

### Gstreamer Type

- other/tensor
- other/tensors (W.I.P. jijoong-moon)

findtype specifics: TBD
meta specifics: W.I.P. jijoong-moon

### Gstreamer Elements (Plugins)

- tensor\_converter
  - Video
    - Prototype for video/xraw (RGB/BGRx) is implemented.
    - Caution: if width is not divisible by 4, RGB video incurs memcpy.
  - Audio
    - Planned
  - Text
    - Planned
- tensor\_filter
  - Main
    - Supported
      - Fixed input/ouput dimensions (fixed by subplugin)
      - Flexible dimensions (output dimension determined by subplugin according to the input dimension determined by pipeline initialization)
      - Invoke subplugin with pre-allocated buffers
      - TODO: Invoke subplugin and let subplugin allocate output buffers.
  - Tensorflow-Lite
    - W.I.P. jinhyuck83-park hello-ahn
  - Custom
    - Supported with example custom subplugins.
  - Other NNFW TBD
- tensor\_sink
  - W.I.P. jy1210-jung
- tensor\_transformer
  - Planned
- tensor\_merge
  - Planned
- tensor\_decode
  - W.I.P. jijoong-moon
- tensor\_source
  - Planned

Note that test elements in /tests/ are not elements for applications. They exist as scaffoldings to test the above elements especially in the case where related elements are not yet implemented.

### Other Components
- CI: up and running. sewon-oh
- Stream test cases: W.I.P. sangjung-woo wook16-song

## Getting Started

### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++
* gstreamer 1.0 and its relatives
* glib 2.0
* cmake >= 2.8

### Linux Self-Hosted Build

**Approach 1.** Build with debian tools

* How to use mk-build-deps
```bash
$ mk-build-deps --install debian/control
$ dpkg -i nnstreamer-build-deps_2018.6.16_all.deb
```
Note that the version name may change. Please check your local directory after excecuting ```mk-build-deps```.

* How to use debuild
```bash
$ debuild
```
If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add ```-uc -us``` options to ```debuild```.


**Approach 2.** Build with Cmake

At the git repo root directory,
```bash
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -rf *        # Ensure the build directory is empty
$ cmake ..
$ make
$ cd ..
```

You may copy the resulting plugin (.so file) to gstreamer plugin repository. Or do
```bash
$ cd build
$ sudo make install
```
if installing NNstreamer plugin libraries into ```%{_libdir}```.


### Clean Build based on Platform

##### Tizen
* https://source.tizen.org/documentation/reference/git-build-system/usage/gbs-build

First install the required packages.
```bash
$ sudo apt install gbs
```

Generates .rpm packages:
```bash
$ gbs build
```
```gbs build``` will execute unit testing as well unlike cmake build.

##### Ubuntu
* https://wiki.ubuntu.com/PbuilderHowto

First install the required packages.
```bash
$ sudo apt install pbuilder debootstrap devscripts
```

Then, create tarball that will contain your chroot environment to build package.
```bash
$ vi ~/.pbuilderrc
# man 5 pbuilderrc
DISTRIBUTION=xenial
OTHERMIRROR="deb http://archive.ubuntu.com/ubuntu xenial universe multiverse"
$ sudo ln -s  ~/.pbuilderrc /root/.pbuilderrc
$ sudo pbuilder create
$ ls -al /var/cache/pbuilder/base.tgz
```

Generates .deb packages:
```bash
$ pdebuild
$ ls -al /var/cache/pbuilder/result/*.deb
```
Note that ```pdebuild``` does not execute unit testing.

## How to Test

- Use the built library as a general gstreamer plugin.

- Unit Test 

For common library unit test
```bash
$ cd build
$ ./unittest_common
```

For all gst-launch-based test cases (mostly golden testing)
```bash
$ cd tests
$ ./testAll.sh
```

## How to write Test Cases
* [How to write Test Cases](Documentation/how-to-write-testcase.md)


## Usage Examples

The implementation is not there yet for using neural network frameworks.

## CI Server
For more details, please access the following web page.
* Press [Here](http://aaci.mooo.com/nnstreamer/ci/standalone/).
