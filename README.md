# NNStreamer

Neural Network Support as Gstreamer Plugins.

NNStreamer is a set of Gstreamer plugins, which allows
Gstreamer developers to adopt neural network models easily and efficiently and
neural network developers to manage stream pipelines and their filters easily and efficiently.

[Architectural Description](https://github.com/nnsuite/nnstreamer/wiki/Architectural-Description) (WIP)

## Objectives

- Provide neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
  - **Efficient Streaming for AI Projects**: Neural network models wanted to use efficient and flexible streaming management as well.
  - **Intelligent Media Filters!**: Use a neural network model as a media filter / converter.
  - **Composite Models!**: Allow to use multiple neural network models in a single stream instance.
  - **Multi Modal Intelligence!**: Allow to use multiple sources for neural network models.

- Provide easy methods to construct media streams with neural network models using the de-facto-standard media stream framework, **GStreamer**.
  - Allow any gstreamer users to put neural network models as if they are media filters.
  - Allow any neural network developers to manage media streams fairly easily.

## Maintainers
* [MyungJoo Ham](https://github.com/myungjoo/)

## Reviewers
* [Jijoong Moon](https://github.com/jijoongmoon)
* [Geunsik Lim](https://github.com/leemgs)
* [Sangjung Woo](https://github.com/again4you)
* [Wook Song](https://github.com/wooksong)
* [Jaeyun Jung](https://github.com/jaeyun-jung)
* [Jinhyuck Park](https://github.com/jinhyuck-park)
* [Hyoungjoo Ahn](https://github.com/helloahn)
* [Sewon Oh](https://github.com/ohsewon)

## Components

Note that this project has just started and many of the components are in design phase.
In [Component Description](Documentation/component-description.md) page, we describe nnstreamer components of the following three categories: data type definitions, gstreamer elements (plugins), and other misc components.

## Getting Started
For more details, please access the following manual.
* Press [Here](Documentation/getting-started.md)

## Test Cases
* Press [Here](Documentation/how-to-use-testcases.md) to read how to write and run Test Cases.

## Usage Examples
- [Example app with camera](nnstreamer_example/example_cam) (stable)
- Wiki page [usage example screenshots](https://github.com/nnsuite/nnstreamer/wiki/usage-examples-screenshots) (stable)

## CI Server

- [CI service status](http://nnsuite.mooo.com/)
- [TAOS-CI config files for nnstreamer](Documentation/ci-config).
