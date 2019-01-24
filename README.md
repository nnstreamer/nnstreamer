# NNStreamer

[![Gitter][gitter-image]][gitter-url]

Neural Network Support as Gstreamer Plugins.

NNStreamer is a set of Gstreamer plugins, which allows
Gstreamer developers to adopt neural network models easily and efficiently and
neural network developers to manage stream pipelines and their filters easily and efficiently.

[Architectural Description](https://github.com/nnsuite/nnstreamer/wiki/Architectural-Description) (WIP)<br />
[NNStreamer: Stream Processing Paradigm for Neural Networks ...](https://arxiv.org/abs/1901.04985) [[pdf](https://arxiv.org/pdf/1901.04985)]<br />
[GStreamer Conference 2018, NNStreamer](https://gstreamer.freedesktop.org/conference/2018/talks-and-speakers.html#nnstreamer-neural-networks-as-filters) [[media](https://github.com/nnsuite/nnstreamer/wiki/Gstreamer-Conference-2018-Presentation-Video)] [[pdf](https://github.com/nnsuite/nnstreamer/wiki/slides/2018_GSTCON_Ham_181026.pdf)]


## Official Releases

Daily Releases

| Arch   | [Tizen](http://download.tizen.org/live/devel%3A/AIC%3A/Tizen%3A/5.0%3A/nnsuite/standard/) | [Ubuntu](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) | Android | Yocto |
| :-- | -- | -- | -- | -- |
|     | 5.5 | 16.04/18.04 | TBD  | TBD |
| arm | Available  | Available  | Planned  | Planned  |
| arm64 | Available  | Available  | N/A  | Planned  |
| x64 | Available  | Available  | N/A  | N/A  |
| x86 | Available  | N/A  | N/A  | N/A  |

[(WIP)](https://github.com/nnsuite/TAOS-CI/issues/452)



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
* [Hyoungjoo Ahn](https://github.com/helloahn)

## Components

Note that this project has just started and many of the components are in design phase.
In [Component Description](Documentation/component-description.md) page, we describe nnstreamer components of the following three categories: data type definitions, gstreamer elements (plugins), and other misc components.

## Getting Started
For more details, please access the following manual.
* Press [Here](Documentation/getting-started.md)

## Usage Examples
- [Example apps](https://github.com/nnsuite/nnstreamer-example) (stable)
- Wiki page [usage example screenshots](https://github.com/nnsuite/nnstreamer/wiki/usage-examples-screenshots) (stable)

## CI Server

- [CI service status](http://nnsuite.mooo.com/)
- [TAOS-CI config files for nnstreamer](.TAOS-CI).


[gitter-url]: https://gitter.im/nnstreamer/Lobby
[gitter-image]: http://img.shields.io/badge/+%20GITTER-JOIN%20CHAT%20%E2%86%92-1DCE73.svg?style=flat-square
