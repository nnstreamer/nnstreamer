# NNStreamer

Neural Network Streamer for AI Projects.
http://suprem.sec.samsung.net/confluence/display/STAR/NN+Streamer

## Objectives

- Provide neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
  - **Efficient Streaming for AI Projects**: Neural network models wanted to use efficient and flexible streaming management as well. 
  - **Intelligent Media Filters!**: Use a neural network model as a media filter / converter.
  - **Composite Models!**: Allow to use multiple neural network models in a single stream instance.
  - **Multi Model Intelligence!**: Allow to use multiple sources for neural network models.

## Components

Note that this project has just started and most of the components are in design phase.

### Gstreamer Type

- other/tensor

findtype specifics: TBD

### Gstreamer Elements (Plugins)

- tensor\_converter
  - Video
    - Prototype for video/xraw (RGB/BGRx) is implemented.
    - Known Issues
      - If video width is not divisible by 4, it is supposed to do zero-padding, which is not working (Found 2018-05-17, #7)
  - Audio
    - Planned
  - Text
    - Planned
- tensor\_filter
  - Main
    - Work in Progress
  - Tensorflow-Lite
    - Skeleton only
  - Custom
    - Concept only
  - Other NNFW TBD
- tensor\_sink
  - Planned
- tensor\_transformer
  - Planned
- tensor\_merge
  - Planned

## How to Build

### Linux Self-Hosted Build


Step 1. Install build-required packages to your system

Approach 1: use mk-build-deps (package name: ```equivs```) / Ubuntu
```
$ mk-build-deps --install debian/control
$ dpkg -i nnstreamer-build-deps_2018.6.16_all.deb
```
Note that the version name may change. Please check your local directory after excecuting ```mk-build-deps```.

Approach 2: try and error / Ubuntu
```
$ debuild
```
If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add ```-uc -us``` options to ```debuild```.

Approach 3: install the following packages (list from Build-Deps in debian/control)
```
gcc, cmake, libgstreamer1.0-dev, libgstreamer-plugins-base1.0-dev, libgstreamer-plugins-good1.0-dev, libglib2.0-dev, libgtest-dev
```
Note that the packages names are from Ubuntu 16.04; however, other Linux distros will have similar names as well.


Step 2. Build with Cmake!

At the git repo root directory,
```
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -Rf *        # Ensure the build directory is empty
$ cmake ..
$ make
$ cd ..
```

You may copy the resulting plugin (.so file) to gstreamer plugin repository. Or do
```
$ cd build
$ sudo make install
```
if installing NNstreamer plugin libraries into ```%{_libdir}```.


### Tizen

Generates .rpm packages:
```
$ gbs build
```
```gbs build``` will execute unit testing as well unlike cmake build.

### Ubuntu

Generates .deb packages:
```
$ pdebuild
```
Note that ```pdebuild``` does not execute unit testing.

## How to Test

- Use the built library as a general gstreamer plugin.

- Unit Test 

For common library unit test
```
$ cd build
$ ./unittest_common
```

For all gst-launch-based test cases (mostly golden testing)
```
$ cd tests
$ ./testAll.sh
```

## [How to write Test Cases](Documentation/how-to-write-testcase.md)

## Usage Examples

The implementation is not there yet for using neural network frameworks.

# CI Server
For more details, please access the following web page.
* Press [Here](http://aaci.mooo.com/nnstreamer/ci/standalone/).
