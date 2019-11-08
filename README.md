# NNStreamer

[![Gitter][gitter-image]][gitter-url] 
[![Code Coverage](http://nnsuite.mooo.com/nnstreamer/ci/badge/codecoverage.svg)](http://nnsuite.mooo.com/nnstreamer/ci/gcov_html/index.html) 
<a href="https://scan.coverity.com/projects/nnsuite-nnstreamer">
<img alt="Coverity Scan Defect Status" src="https://img.shields.io/endpoint?url=https://nnsuite.mooo.com/nnstreamer/ci/badge/badge_coverity.json" />
</a> 
![GitHub repo size](https://img.shields.io/github/repo-size/nnsuite/nnstreamer) 
![GitHub issues](https://img.shields.io/github/issues/nnsuite/nnstreamer) 
![GitHub pull requests](https://img.shields.io/github/issues-pr/nnsuite/nnstreamer) 

Neural Network Support as Gstreamer Plugins.

NNStreamer is a set of Gstreamer plugins that allow
Gstreamer developers to adopt neural network models easily and efficiently and
neural network developers to manage stream pipelines and their filters easily and efficiently.

[Architectural Description](https://github.com/nnsuite/nnstreamer/wiki/Architectural-Description) (WIP)<br /> <br />

[NNStreamer: Stream Processing Paradigm for Neural Networks ...](https://arxiv.org/abs/1901.04985) [[pdf/tech report](https://arxiv.org/pdf/1901.04985)]<br />
[GStreamer Conference 2018, NNStreamer](https://gstreamer.freedesktop.org/conference/2018/talks-and-speakers.html#nnstreamer-neural-networks-as-filters) [[media](https://github.com/nnsuite/nnstreamer/wiki/Gstreamer-Conference-2018-Presentation-Video)] [[pdf/slides](https://github.com/nnsuite/nnstreamer/wiki/slides/2018_GSTCON_Ham_181026.pdf)]<br />
[Naver Tech Talk (Korean)](https://www.facebook.com/naverengineering/posts/2255360384531425) [[media](https://youtu.be/XvXxcnbRjgU)] [[pdf/slides](https://www.slideshare.net/NaverEngineering/nnstreamer-stream-pipeline-for-arbitrary-neural-networks)]<br />
[ResearchGate Page of NNStreamer](https://www.researchgate.net/project/Neural-Network-Streamer-nnstreamer)


## Official Releases

| Arch   | [Tizen](http://download.tizen.org/live/devel%3A/AIC%3A/Tizen%3A/5.0%3A/nnsuite/standard/) | [Ubuntu](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) | Android/NDK Build | Android/APK | Yocto | macOS |
| :-- | -- | -- | -- | -- | -- | -- |
|     | 5.5 | 16.04/18.04 | 7.0/N | 7.0/N | TBD |   |
| arm | Available  | Available  | Ready | Available| Ready | N/A |
| arm64 | Available  | Available  | Ready | Available | Planned | N/A |
| x64 | Available  | Available  | Ready  | Ready | Ready | Available |
| x86 | Available  | N/A  | N/A  | N/A  | N/A | N/A |
|     | [Tizen Repo](http://download.tizen.org/snapshots/tizen/unified/latest/repos/standard/packages/) | [PPA](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa) |   | [JCenter](https://bintray.com/beta/#/nnsuite/nnstreamer/nnstreamer?tab=overview) |   | [Brew Tap](Documentation/getting-started-macos.md) |

- Ready: CI system ensures build-ability and unit-testing. Users may easily build and execute. However, we do not have automated release & deployment system for this instance.
- Available: binary packages are released and deployed automatically and periodically along with CI tests.
- Daily Release: [(WIP)](https://github.com/nnsuite/TAOS-CI/issues/452)
- SDK Support: Tizen Studio (5.5 M2+) / Android Studio (JCenter, "nnstreamer")



## Objectives

- Provide neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
  - **Efficient Streaming for AI Projects**: Apply efficient and flexible stream pipeline to neural networks.
  - **Intelligent Media Filters!**: Use a neural network model as a media filter / converter.
  - **Composite Models!**: Multiple neural network models in a single stream pipeline instance.
  - **Multi Modal Intelligence!**: Multiple sources and stream paths for neural network models.

- Provide easy methods to construct media streams with neural network models using the de-facto-standard media stream framework, **GStreamer**.
  - Gstreamer users: use neural network models as if they are yet another media filters.
  - Neural network developers: manage media streams easily and efficiently.

## Maintainers
* [MyungJoo Ham](https://github.com/myungjoo/)

## Reviewers
* [Jijoong Moon](https://github.com/jijoongmoon)
* [Geunsik Lim](https://github.com/leemgs)
* [Sangjung Woo](https://github.com/again4you)
* [Wook Song](https://github.com/wooksong)
* [Jaeyun Jung](https://github.com/jaeyun-jung)
* [Hyoungjoo Ahn](https://github.com/helloahn)
* [Parichay Kapoor](https://github.com/kparichay)
* [Dongju Chae](https://github.com/dongju-chae)

## Components

Note that this project has just started and many of the components are in design phase.
In [Component Description](Documentation/component-description.md) page, we describe nnstreamer components of the following three categories: data type definitions, gstreamer elements (plugins), and other misc components.

## Getting Started
For more details, please access the following manuals.
* For Linux-like systems such as Tizen, Debian, and Ubuntu, press [here](Documentation/getting-started.md).
* For macOS systems, press [here](Documentation/getting-started-macos.md).
* To build an API library for Android, press [here](api/android/README.md).

## Usage Examples
- [Example apps](https://github.com/nnsuite/nnstreamer-example) (stable)
- Wiki page [usage example screenshots](https://github.com/nnsuite/nnstreamer/wiki/usage-examples-screenshots) (stable)

## CI Server

- [CI service status](http://nnsuite.mooo.com/)
- [TAOS-CI config files for nnstreamer](.TAOS-CI).


[gitter-url]: https://gitter.im/nnstreamer/Lobby
[gitter-image]: http://img.shields.io/badge/+%20GITTER-JOIN%20CHAT%20%E2%86%92-1DCE73.svg?style=flat-square
