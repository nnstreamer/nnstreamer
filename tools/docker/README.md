---
title: Docker Support
...

# Getting Started with Docker

## Prerequisites

In order to build and run Docker containers from the provided Dockerfile, [docker-engine](https://docs.docker.com/engine/) is required. Before you get started, make sure you have installed docker-engine in your **Ubuntu** machine. If you need to install it, refer [the official instructions](https://docs.docker.com/engine/install/ubuntu/#installation-methods). Note that the provided Dockerfile is tested with Ubuntu distros.

## Building Docker images

The `docker build` command builds the Docker image from the given Dockerfile in the specified `PATH`. For example, the following commands create a container image using the Dockerfile in the current directory.

```bash
$ cd tools/docker
$ docker build .
```

Instead of `.`, it is possible to specify the the directory that includes the Dockerfile.

```bash
$ pwd
~/nnstreamer
$ docker build ./tools/docker/
```

To check the created image, use the `images` sub-command.

```bash
$ docker images
REPOSITORY                     TAG        IMAGE ID       CREATED         SIZE
<none>                         <none>     6f207865d65b   8 minutes ago   563MB
...
```

By default, the created image is based on Ubuntu 18.04 and all of the sub-plugins available for Ubuntu 18.04 are installed into the image. It will be addressed later section.

To name and tag the created image, use the `-t` option with the [`build`](https://docs.docker.com/engine/reference/commandline/build/) sub-command.

```bash
$ cd tools/docker
$ docker build . -t nns:latest
...
$ docker images
REPOSITORY                     TAG        IMAGE ID       CREATED          SIZE
nns                            latest     6f207865d65b   34 minutes ago   563MB
...
```

## Running the Docker image

To use NNStreamer installed in the container image, use the [`run`](https://docs.docker.com/engine/reference/commandline/run/) sub-command. With this sub-command, it is possible to run the pipeline using `gst-launch-1.0` or custom binaries that rely on the nnstreamer libraries. In the following example, `/bin/bash` will be given for the command to run for the demonstration purpose.

```bash
$ docker images
REPOSITORY                     TAG        IMAGE ID       CREATED          SIZE
nns                            latest     6f207865d65b   34 minutes ago   563MB
...
$ docker run -it nns:latest /bin/bash
nns@a7c5ba046c0e:~$
```

After the above `docker run` command, you can see the command line interface provided by `/bin/bash`. In this context, it is possible to use the GStreamer/NNStreamer tools as follows:

```bash
nns@e80a744f93ce:~$ gst-inspect-1.0 nnstreamer
Plugin Details:
  Name                     nnstreamer
  Description              nnstreamer plugin library
  Filename                 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libnnstreamer.so
  Version                  2.1.0
  License                  LGPL
  Source module            nnstreamer
  Binary package           nnstreamer
  Origin URL               https://github.com/nnstreamer/nnstreamer

  tensor_src_iio: TensorSrcIIO
  tensor_query_client: TensorQueryClient
  tensor_query_serversink: TensorQueryServerSink
  tensor_query_serversrc: TensorQueryServerSrc
  tensor_rate: TensorRate
  tensor_if: TensorIf
  tensor_transform: TensorTransform
  tensor_split: TensorSplit
  tensor_sparse_dec: TensorSparseDec
  tensor_sparse_enc: TensorSparseEnc
  tensor_sink: TensorSink
  tensor_reposrc: TensorRepoSrc
  tensor_reposink: TensorRepoSink
  tensor_mux: TensorMux
  tensor_merge: TensorMerge
  tensor_filter: TensorFilter
  tensor_demux: TensorDemux
  tensor_decoder: TensorDecoder
  tensor_crop: TensorCrop
  tensor_converter: TensorConverter
  tensor_aggregator: TensorAggregator

  21 features:
  +-- 21 elements
```
