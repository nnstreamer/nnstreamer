FROM ubuntu:18.04

RUN apt update
RUN apt install -y software-properties-common
RUN apt-add-repository ppa:nnstreamer
RUN apt update

RUN apt install -y git
RUN git clone https://github.com/nnstreamer/nnstreamer.git /opt/nnstreamer

RUN apt install -y \
    meson \
    ninja-build \
    build-essential \
    pkg-config \
    cmake \
    libglib2.0-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    libgstreamer-plugins-base1.0-dev \
    libgtest-dev

WORKDIR /opt/nnstreamer/
RUN meson build
RUN ninja -C build install
ENV GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0/
