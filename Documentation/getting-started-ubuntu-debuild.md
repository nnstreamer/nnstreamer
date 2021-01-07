---
title: Ubuntu Debuild/Pdebuild
...

## Getting Started: Ubuntu Debuild/Pdebuild

In order to control how pdebuild/debuild build, you need to edit files in ```${nnstreamer-source}/debian/```.


### Pdebuild, a sandboxed build environment. (Ubuntu 16.04, 18.04)


This guide uses the nnstreamer PPA to resolve additional build-dependencies (e.g., tensorflow/tensorflow-lite 1.13).

Install build tools for pdebuild:
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

Then, you may install the resulting .deb files to your system.


Refer to [PbuilderHowto](https://wiki.ubuntu.com/PbuilderHowto) for more about pdebuild.



### Debuild, build with system libraries. (Ubuntu/Debian)

After installing all the required packages to your system, you may run ```debuild``` to get .deb packages.

In Ubuntu, you may prepare your system easily with the following commands, which installs prebuilt binaries from nnstreamer's PPA repository.
```bash
$ sudo apt-add-repository ppa:nnstreamer
$ sudo apt install ssat tensorflow-dev tensorflow-lite-dev libprotobuf-dev # you may add pytorch and other libraries, too
```

Note that ssat is required for unit testing. You may download it at [Github SSAT](https://github.com/myungjoo/SSAT).

Refer to [debuild command](https://www.debian.org/doc/manuals/maint-guide/build.en.html#debuild) for how to use ```debuild```.


***1. Clone the needed repositories***

You may skip this if you have downloaded and installed PPA packages of SSAT and tensorflow.

```bash
$ git clone https://github.com/myungjoo/SSAT ssat
$ git clone https://git.tizen.org/cgit/platform/upstream/tensorflow
$ git clone https://github.com/nnstreamer/nnstreamer
```

***2. Fix tensorflow build errors***

You may skip this if you have downloaded and installed PPA packages of SSAT and tensorflow.

A script, ```tensorflow/contrib/lite/Makefile```, may fail, depending your shell.
Replace the ARCH macro (HOST_ARCH in some versions):

From:
```makefile
ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)
```

To:
```makefile
ARCH := $(shell uname -m | sed -e 's/i[3-8]86/x86_32/')
```



***3. Prepare for debuild (installing required dependencies).***

Skip ssat and tensorflow if you have already installed them.

```bash
$ for i in ssat tensorflow nnstreamer; do \
  (cd $i && sudo mk-build-deps --install debian/control && sudo dpkg -i *.deb || break); \
  done
```

***4. Run debuild and get .deb packages.***

Skip ssat and tensorflow if you have already installed them.

```bash
$ export DEB_BUILD_OPTIONS="parallel=$(($(cat /proc/cpuinfo |grep processor|wc -l) + 1))"
$ for i in ssat tensorflow nnstreamer; do \
  (cd $i && time debuild -us -uc || break); \
  done
```

If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add ```-uc -us``` options to ```debuild```.

***5. Install the generated .deb files.***

Skip ssat and tensorflow if you have already installed them.

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
