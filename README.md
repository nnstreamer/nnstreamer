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


```
$ gbs build
```
```gbs build``` will execute unit testing as well unlike cmake build.

## How to Test

- Use the built library as a general gstreamer plugin.

- Unit Test 

For common library unit test
```
$ cd build
$ ./unittest_common
```

For tensor_converter golden test
```
$ cd tensor_convert/test
$ ./runTest.sh
```

## Usage Examples

The implementation is not there yet for using neural network frameworks.
