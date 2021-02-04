---
title: Tizen GBS
...

## Getting Started: Tizen GBS

* If you are a Tizen platform developer, you may use GBS to build your customized nnstreamer for your developmental requirements.

* If you are a Tizen application developer, you may still use GBS to build your customized nnstreamer for your own Tizen devices. However, you won't be able to publish Tizen application that depends on your modification. You need to upstream your modifications to github.com/nnstreamer/nnstreamer and wait for next Tizen updates.

[How to install GBS](https://source.tizen.org/documentation/developer-guide/getting-started-guide/installing-development-tools)

[How to build with GBS](https://source.tizen.org/documentation/reference/git-build-system/usage/gbs-build)


### Build without options

```bash
$ gbs -c .TAOS-CI/.gbs.conf build
```

### Build with options

***Enable full unit testing***
```bash
$ gbs -c .TAOS-CI/.gbs.conf build --define "unit_test 1"
```

***Get unit-test coverage report with the unit test for aarch64***
```bash
$ gbs -c .TAOS-CI/.gbs.conf build -A aarch64 --define "unit_test 1" --define "testcoverage 1"
```

***Update Tizen-build options and build it without git commit***
```bash
$ vi packaging/nnstreamer.spec
# Modify the default options described in the top lines
$ gbs -c .TAOS-CI/.gbs.conf build --include-all
# Build with changes not committed.
```

### Install GBS-built packages to target devices

Depending on your GBS configuration (usually at ```~/.gbs.conf```), the built RPM files are located at ```${GBS-ROOT}/local/repos/tizen/${ARCH}/RPMS/```.

Copy the RPM files to your devices with SDB Tizen tool.
```bash
$ sdb root on
$ sdb push LOCAL_RPM_PATH.rpm /DEVICE/PATH/
$ sdb shell
TIZEN$ cd /DEVICE/PATH/
TIZEN$ rpm -U filename.rpm
```

You will need root permission and your device should be unlocked or SMACK should be disabled. You may also need to remount the root partition as read-write.

### Additional materials for Tizen developers

[Writing Tizen Native Apps with NNStreamer / ML APIs](Documentation/writing-tizen-native-apps.md)
