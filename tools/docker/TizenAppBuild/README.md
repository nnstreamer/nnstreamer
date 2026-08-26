---
title: Tizen App Build Docker Support
...

# Tizen App Build Docker Image

This directory provides a docker image with the [Tizen Studio](https://docs.tizen.org/application/tizen-studio/) CLI tools so that Tizen native applications can be build-tested without a local Tizen Studio installation (e.g., in github-action CI jobs).

The image installs the Tizen Studio web CLI and the `TIZEN-10.0-NativeAppDevelopment-CLI` package set. The versions can be overridden with the `TIZEN_STUDIO_VERSION` and `TIZEN_PLATFORM_PKG` build arguments. The installer download is verified against the `TIZEN_STUDIO_SHA256` build argument, which must be updated in lockstep with `TIZEN_STUDIO_VERSION`: download the installer for the new version, compute its `sha256sum`, and pass (or commit) it together with the version. A checksum mismatch also occurs if Tizen re-spins the installer under the same URL; verify the new binary and refresh the pin in that case.

## Building the image

```bash
$ docker build -t nnstreamer/tizen-app-build tools/docker/TizenAppBuild
```

## Verifying the toolchain

`tizen-build-test.sh` creates a native template project, replaces its source with a minimal nnstreamer C-API consumer, and builds it. This verifies the installed toolchain, the platform rootstrap, and the ML API headers/libraries that nnstreamer-based Tizen apps depend on. It does not require signing certificates.

```bash
$ docker run --rm nnstreamer/tizen-app-build ./tizen-build-test.sh
```

Two optional environment variables extend the check, which the `docker_build_test.yml` CI workflow uses to verify that changes in this repository do not break Tizen apps:

- `NNS_RPM_DIR`: a directory (mounted into the container) with nnstreamer devel RPMs built from this repository via GBS. Their headers are overlaid onto the platform rootstrap and the test app additionally compiles against them.
- `NNS_EXAMPLE_DIR`: a Tizen native app project directory (e.g., `Tizen.native/PipelineSample` from [nnstreamer-example](https://github.com/nnstreamer/nnstreamer-example)). The project is retargeted to the image's platform profile and built.

To build your own application sources, mount them into the container and use the `tizen` CLI. The container runs as the `sdk` user (uid 1000); make sure the mounted sources are readable (and their output directory writable) by that uid:

```bash
$ docker run --rm -it -v /path/to/app:/home/sdk/app nnstreamer/tizen-app-build
sdk@container:~$ cd app && tizen build-native -a arm -c gcc -C Debug
```
