## Getting Started

### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++
* gstreamer 1.0 and its relatives
* glib 2.0
* meson >= 0.50

### Install via PPA repository (Debian/Ubuntu)

The nnstreamer releases are at a PPA repository. In order to install it, use:

```bash
$ sudo apt-add-repository ppa:nnstreamer
$ sudo apt install nnstreamer
```

### Clean build with pdebuild (Ubuntu 16.04/18.04)

Use the nnstreamer PPA to resolve additional build-dependencies (tensorflow/tensorflow-lite).

Install build tools if needed:
```bash
$ sudo apt install pbuilder debootstrap devscripts
```

The following example configuration is for Ubuntu 16.04:
```bash
$ cat ~/.pbuilderrc
DISTRIBUTION=xenial
COMPONENTS="main restricted universe multiverse"
OTHERMIRROR="deb http://archive.ubuntu.com/ubuntu xenial main restricted universe multiverse |\
  deb http://archive.ubuntu.com/ubuntu xenial-security main restricted universe multiverse |\
  deb http://archive.ubuntu.com/ubuntu xenial-updates main restricted universe multiverse |\
  deb [trusted=yes] http://ppa.launchpad.net/nnstreamer/ppa/ubuntu xenial main"
$ sudo ln -s  ~/.pbuilderrc /root/.pbuilderrc
$ sudo pbuilder create
```

Run pdebuild to build and get the package.
```bash
$ pdebuild
$ ls -al /var/cache/pbuilder/result/*.deb
```

Refer to [PbuilderHowto](https://wiki.ubuntu.com/PbuilderHowto) for more about pdebuild.



### Linux Self-Hosted Build

**Approach 1.** Build with Debian/Ubuntu tools

***Clone the needed repositories***

```bash
$ git clone https://github.com/myungjoo/SSAT ssat
$ git clone https://git.tizen.org/cgit/platform/upstream/tensorflow
$ git clone https://github.com/nnstreamer/nnstreamer
```

Alternatively, you may simply download binary packages from PPA (ssat and tensorflow):
```bash
$ sudo apt-add-repository ppa:nnstreamer
$ sudo apt install ssat tensorflow-dev tensorflow-lite-dev libprotobuf-dev
```


***Fix tensorflow for it to build properly***

You may skip this if you have downloaded binary packages from PPA.

There is a shell script call at tensorflow/contrib/lite/Makefile that may
fail, depending on the shell you're using. The best is to replace
the ARCH detection macro (it is named renamed to HOST_ARCH on tensorflow
upstream) from:

```makefile
ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)
```

to:

```makefile
ARCH := $(shell uname -m | sed -e 's/i[3-8]86/x86_32/')
```

***Build .deb packages***

Installing required dependencies:

```bash
$ for i in ssat tensorflow nnstreamer; do \
  (cd $i && sudo mk-build-deps --install debian/control && sudo dpkg -i *.deb || break); \
  done
```

Creating the .deb packages:

```bash
$ export DEB_BUILD_OPTIONS="parallel=$(($(cat /proc/cpuinfo |grep processor|wc -l) + 1))"
$ for i in ssat tensorflow nnstreamer; do \
  (cd $i && time debuild -us -uc || break); \
  done
```

If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add ```-uc -us``` options to ```debuild```.

***Install the generated \*.deb files***

The files will be there at the parent dir. E. g. at nnbuilder/.. directory.

In order to install them (should run as root):

```bash
$ sudo apt install ./ssat_*.deb ./tensorflow-lite-dev_*.deb ./tensorflow-dev_*.deb
$ sudo apt install ./nnstreamer_0.1.0-1rc1_amd64.deb
```

If you need nnstreamer development package:

```bash
#apt install ./nnstreamer-dev_0.1.0-1rc1_amd64.deb
```

**Approach 2.** Build with meson
* https://mesonbuild.com/Getting-meson.html

Install the required packages.

```bash
$ sudo apt install meson ninja-build
```

Build at the git repo root directory, this will install nnstreamer plugins and related files.

```bash
$ meson build
$ ninja -C build install
```

- Installed nnstreamer plugins to ```{prefix}/{libdir}/gstreamer-1.0```
- Installed subplugins and libraries to ```{prefix}/{libdir}```
- Installed common header files to ```{prefix}/{includedir}```


### Tizen
* https://source.tizen.org/documentation/reference/git-build-system/usage/gbs-build

First install the required packages.
```bash
$ sudo apt install gbs
```

Generates .rpm packages:
```bash
$ gbs build
```
```gbs build``` will execute unit testing as well unlike cmake build.

- [Writing Tizen Native Apps with NNStreamer / ML APIs](writing-tizen-native-apps.md)
