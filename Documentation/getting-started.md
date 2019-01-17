## Getting Started

### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++
* gstreamer 1.0 and its relatives
* glib 2.0
* meson >= 0.40

### Install via PPA repository (Debian/Ubuntu)

The nnstreamer releases are at a PPA repository. In order to install it, use:

```bash
$ sudo apt-add-repository ppa:nnstreamer
$ sudo apt install nnstreamer
```

### Linux Self-Hosted Build

**Approach 1.** Build with Debian/Ubuntu tools

***Clone the needed repositories***

```bash
$ git clone https://github.com/myungjoo/SSAT ssat
$ git clone https://git.tizen.org/cgit/platform/upstream/tensorflow
$ git clone https://github.com/nnsuite/nnstreamer
```

***Fix tensorflow for it to build properly***

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

Installing required depencencies:

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

Install the required packages. (meson >= 0.40)

```bash
$ sudo apt install meson ninja-build
```

Build at the git repo root directory, this will install nnstreamer plugins and related files.

```bash
$ meson --werror build
$ ninja -C build install
```

- Installed nnstreamer plugins to ```{prefix}/{libdir}/gstreamer-1.0```
- Installed subplugins and libraires to ```{prefix}/{libdir}```
- Installed common header files to ```{prefix}/{includedir}```


### Clean Build based on Platform

##### Tizen
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

##### Ubuntu
* https://wiki.ubuntu.com/PbuilderHowto

First install the required packages.
```bash
$ sudo apt install pbuilder debootstrap devscripts
```

Then, create tarball that will contain your chroot environment to build package. (for Ubuntu 16.04)
```bash
$ vi ~/.pbuilderrc
# man 5 pbuilderrc
DISTRIBUTION=xenial
OTHERMIRROR="deb http://archive.ubuntu.com/ubuntu xenial universe multiverse |deb [trusted=yes] http://ppa.launchpad.net/nnstreamer/ppa/ubuntu xenial main"
$ sudo ln -s  ~/.pbuilderrc /root/.pbuilderrc
$ sudo pbuilder create
```
Because Ubuntu 16.04 does not have tensorflow-lite-dev in its repository, you need to add
a PPA repository or SPIN/OBS repository of TAOS:UbuntuTools.

Generates .deb packages:
```bash
$ pdebuild
$ ls -al /var/cache/pbuilder/result/*.deb
```
Note that ```pdebuild``` does not execute unit testing.
