# NNStreamer
Neural Network Streamer for AI Projects. http://suprem.sec.samsung.net/confluence/display/SRAMLABC/NN+Streamer

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

### Gstreamer Elements (Plugins)

- tensor\_converter
  - Video
    - Prototype for video/xraw (RGB/BGRx) is implemented.
    - Known Issues
      - If video width is not divisible by 4, it is supposed to do zero-padding, which is not working (Found 2018-05-17, #7)
  - Audio
    - TBD
  - Text
    - TBD
- tensor\_transformer
- tensor\_filter
- tensor\_sink
- tensor\_merge

## How to Build

### Linux Self-Hosted Build

In each plugin source directory,
```
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -Rf *        # Ensure the build directory is empty
$ cmake ..
$ make
```

You may copy the resulting plugin (.so file) to gstreamer plugin repository.


### Tizen

Not-Yet-Implemented Feature
Tizen is going to be the first official target along with Ubuntu.

```
$ gbs build
```

## How to Test

Use the built library as a general gstreamer plugin.

## Usage Examples

The implementation is not there yet for using neural network frameworks.
