FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TIZEN_STUDIO_VERSION=6.1
ARG TIZEN_STUDIO_URL=https://download.tizen.org/sdk/Installer/tizen-studio_${TIZEN_STUDIO_VERSION}/web-cli_Tizen_Studio_${TIZEN_STUDIO_VERSION}_ubuntu-64.bin
# Update together with TIZEN_STUDIO_VERSION. Upstream publishes no checksums;
# this pins the binary observed at the time the version was chosen.
ARG TIZEN_STUDIO_SHA256=7f33fe09eda4eb5d50463cf997e8f676d455060720470070ae1932994413aaf4
ARG TIZEN_PLATFORM_PKG=TIZEN-10.0-NativeAppDevelopment-CLI
ENV LC_ALL=C.UTF-8

# python3.8 is a prerequisite of the Tizen Studio platform packages but is
# not available in the Ubuntu 22.04 archive; use the deadsnakes PPA.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget git make zip unzip \
        pciutils rpm2cpio cpio gettext openjdk-11-jre-headless \
        python3 python3.8 libpython3.8 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 sdk && \
    useradd --uid 1000 --gid sdk --shell /bin/bash --create-home sdk

USER sdk
ENV HOME=/home/sdk
WORKDIR /home/sdk

RUN curl -fsSL "${TIZEN_STUDIO_URL}" -o /tmp/tizen-studio-cli.bin && \
    echo "${TIZEN_STUDIO_SHA256}  /tmp/tizen-studio-cli.bin" | sha256sum -c - && \
    chmod +x /tmp/tizen-studio-cli.bin && \
    /tmp/tizen-studio-cli.bin --accept-license "${HOME}/tizen-studio" && \
    rm -f /tmp/tizen-studio-cli.bin

RUN "${HOME}/tizen-studio/package-manager/package-manager-cli.bin" install \
        --accept-license "${TIZEN_PLATFORM_PKG}" && \
    rm -rf "${HOME}/tizen-studio-data/cache"/* "${HOME}/tizen-studio-data/temp"/*

ENV PATH="/home/sdk/tizen-studio/tools/ide/bin:/home/sdk/tizen-studio/tools:${PATH}"

COPY --chown=sdk:sdk tizen-build-test.sh /home/sdk/tizen-build-test.sh
RUN chmod +x /home/sdk/tizen-build-test.sh

CMD ["/bin/bash"]
