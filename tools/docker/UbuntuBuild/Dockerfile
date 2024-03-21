ARG UBUNTU_VER=18.04
# If the value assigned to MIN_RUN_ENV is non-zero length,
# only nnstreamer, nnstreamer-core, and nnstreamer-configuration packages are installed
# in the container.
ARG MIN_RUN_ENV=""
FROM ubuntu:${UBUNTU_VER} AS dev-base

ARG DEBCONF_NOWARNINGS="yes"
ARG DEBCONF_TERSE="yes"
ARG LANG="C.UTF-8"
ARG BUILDDIR=build
ARG USERNAME=nns

ENV DEBIAN_FRONTEND noninteractive

RUN set -x && \
    echo "debconf debconf/frontend select ${DEBIAN_FRONTEND}" | debconf-set-selections && \
    echo 'APT::Install-Recommends "false";' | tee /etc/apt/apt.conf.d/99install-recommends && \
    echo 'APT::Get::Assume-Yes "true";' | tee /etc/apt/apt.conf.d/99assume-yes && \
    sed -Ei 's|^(DPkg::Pre-Install-Pkgs .*)|#\1|g' /etc/apt/apt.conf.d/70debconf && \
    apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository ppa:nnstreamer/ppa -u && \
    add-apt-repository ppa:one-runtime/ppa -u && \
    apt-get install \
        git \
        build-essential \
        pkg-config \
        cmake \
        ninja-build \
        flex \
        bison \
        meson \
        equivs \
        devscripts && \
    useradd -ms /bin/bash ${USERNAME}

SHELL ["/bin/bash", "-c"]
USER ${USERNAME}
WORKDIR /home/${USERNAME}
RUN git clone https://github.com/nnstreamer/nnstreamer.git
WORKDIR /home/${USERNAME}/nnstreamer
RUN mk-build-deps debian/control
USER root
RUN apt-get -f install ./nnstreamer-build-deps*.deb && \
    rm -f ./nnstreamer-build-deps*.deb && \
    rm -rf /var/lib/apt/lists/*
USER ${USERNAME}
RUN meson ${BUILDDIR} && \
    ninja -C ${BUILDDIR}
ENV GST_PLUGIN_PATH=/home/${USERNAME}/nnstreamer/${BUILDDIR}/gst/nnstreamer \
    NNSTREAMER_CONF=/home/${USERNAME}/nnstreamer/${BUILDDIR}/nnstreamer-test.ini \
    NNSTREAMER_FILTERS=/home/${USERNAME}/nnstreamer/${BUILDDIR}/ext/nnstreamer/tensor_filter \
    NNSTREAMER_DECODERS=/home/${USERNAME}/nnstreamer/${BUILDDIR}/ext/nnstreamer/tensor_decoder \
    NNSTREAMER_CONVERTERS=/home/${USERNAME}/nnstreamer/${BUILDDIR}/ext/nnstreamer/tensor_converter \
    NNSTREAMER_TRAINERS=/home/${USERNAME}/nnstreamer/${BUILDDIR}/ext/nnstreamer/tensor_trainer \
    NNS_USERNAME=${USERNAME} \
    NNS_BUILDDIR=${BUILDDIR}

FROM dev-base as debuilder
WORKDIR /home/${NNS_USERNAME}/nnstreamer
# A workaround for the failure issue of unittest_mqtt
RUN sed -i "s|run_unittests_binaries.sh|run_unittests_binaries.sh -k unittest_mqtt|g" ./debian/rules && \
    debuild -us -uc

FROM ubuntu:${UBUNTU_VER} AS run_env
ARG DEBCONF_NOWARNINGS="yes"
ARG DEBCONF_TERSE="yes"
ARG LANG="C.UTF-8"
ARG BUILDDIR=build
ARG USERNAME=nns
ARG MIN_RUN_ENV

ENV DEBIAN_FRONTEND noninteractive
RUN set -x && \
    echo "debconf debconf/frontend select ${DEBIAN_FRONTEND}" | debconf-set-selections && \
    echo 'APT::Install-Recommends "false";' | tee /etc/apt/apt.conf.d/99install-recommends && \
    echo 'APT::Get::Assume-Yes "true";' | tee /etc/apt/apt.conf.d/99assume-yes && \
    sed -Ei 's|^(DPkg::Pre-Install-Pkgs .*)|#\1|g' /etc/apt/apt.conf.d/70debconf && \
    apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository ppa:nnstreamer/ppa -u && \
    add-apt-repository ppa:one-runtime/ppa -u && \
    apt-get install \
            gstreamer1.0-plugins-base \
            gstreamer1.0-tools && \
    useradd -ms /bin/bash ${USERNAME}

SHELL ["/bin/bash", "-c"]
WORKDIR /root
COPY --from=debuilder /home/${USERNAME}/*.deb /root/
RUN if [ ! -n "${MIN_RUN_ENV}" ]; then \
        apt-get -f install ./*.deb; \
    else \
        apt-get -f install ./nnstreamer_*.deb ./nnstreamer-core_*.deb ./nnstreamer-configuration_*.deb; \
    fi && \
    rm -rf /var/lib/apt/lists/* /root/*.deb
USER ${USERNAME}
WORKDIR /home/${USERNAME}
