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
* Jijoong Moon (jijoon.moon@samsung.com)
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
- tensormux
  - Supported
    - Mux mutiple "other/tensor"s and make "other/tensors"
    - Currently maximum number of tensor to be muxed is 16
- tensordemux
  - Supported
    - Demux "other/tensors' to "other/tensor"s
    - tensor to be pushed to downstream can be choosen with "tensorpick" option.
- tensor\_source
  - Planned

Note that test elements in /tests/ are not elements for applications. They exist as scaffoldings to test the above elements especially in the case where related elements are not yet implemented.

### Other Components
- CI: up and running. sewon-oh
- Stream test cases: W.I.P. sangjung-woo wook16-song

## Getting Started
For more details, please access the following manual.
* Press [Here](Documentation/getting-started.md)

## Test Cases
* Press [Here](Documentation/how-to-use-testcases.md) to read how to write and run Test Cases.

## Usage Examples

The implementation is not there yet for using neural network frameworks.

## CI Server
For more details, please access the following web page.
* Press [Here](http://aaci.mooo.com/nnstreamer/ci/standalone/).

TAOS-CI config files for nnstreamer
* Press [Here](http://github.sec.samsung.net/STAR/nnstreamer/tree/tizen/Documentation/ci-config).
