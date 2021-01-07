---
title: Meson/Ninja Build
...

## Getting Started: Meson/Ninja Build

NNStreamer uses meson/ninja as its standard build environment; both Tizen(GBS-OBS/.rpm) and Ubuntu(pdebuild/.deb) builds uses meson/ninja.

With meson/ninja, you may build in the git repo directory and install nnstreamer binaries to your designated directories.
However, you should be careful on configuring nnstreamer; i.e., the paths of subplugins and .ini file locations. Please note that nnstreamer tries to load /etc/nnstreamer.ini by default.

* https://mesonbuild.com/Getting-meson.html


### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++ (C++14 if you want C++-class as filters)
* gstreamer 1.0 and its relatives
* glib 2.0
* meson >= 0.50
* ninja-build
* Neural network frameworks or libraries for plugins (e.g., tensorflow) you want to use, including their pkgconfig files or mechanisms to support meson to discover the corresponding frameworks. If you use our packages for tensorflow/pytorch, you do not need to worry.
    * Possible frameworks for "extra" plugins: tensorflow, pytorch, protobuf, flatbuf, openvino, ncsdk2, Verisilicon-Vivante, SNPE, ...
* [SSAT](https://github.com/myungjoo/SSAT) (optional. required for unit testing)
* gtest (optional. required for unit testing)

The minimal requirement to build nnstreamer with default configuration is
```bash
$ sudo apt install meson ninja-build gcc g++ libglib2.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

***1. Clone the needed repositories***

```bash
$ git clone https://github.com/myungjoo/SSAT ssat
$ git clone https://git.tizen.org/cgit/platform/upstream/tensorflow
$ git clone https://github.com/nnstreamer/nnstreamer
```

***2. Fix tensorflow build errors***

You may skip this if you have downloaded binary packages from PPA.

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

***3. Configure the build***

Configure build in ```./build``` directory with default options.
```bash
$ meson build
```

For possible configurations, please refer to [meson_options.txt](https://github.com/nnstreamer/nnstreamer/blob/main/meson_options.txt).
If you want to designate nnstreamer's binary locations, provide such with ```--prefix```, ```--sysconfdir```, ```--libdir```, ```--bindir```, and ```--includedir```.
For example, in order to install as general Linux system libraries you may configure:
```bash
$ meson build ${whateveroptionyouwant....} --prefix=/ --sysconfdir=/etc --libdir=/usr/lib --bindir=/usr/bin --includedir=/usr/include
```

Documentation for possible meson options: WIP


***4. Build***

Assuming you have configured meson at ```./build```.
```bash
$ ninja -C build
```


***5. Install***

Assuming you have configured meson at ```./build```.
```bash
$ ninja -C build install
```

Then, it will install:
- nnstreamer plugins to ```{libdir}/gstreamer-1.0/```
- libraries to ```{libdir}/```
- subplugins to ```{prefix}/lib/nnstreamer/PLUGIN-TYPE/```
- common header files to ```{includedir}/```


***6. Verifying installation***

Unless you have given sysconfdir (```--sysconfdir``` meson option), ```/etc/nnstreamer.ini``` is the default configuration file. If ```--sysconfdir``` was given at build time, the default configuration file is at {sysconfdir}/nnstreamer.ini.

Unless overriden by envvars, the paths in the .ini file are used to search for subplugins. Therefore, if a subplugin (e.g., tensorflow-lite filter) does not work, check the subplugin locations and their configurations first.

We will supply utility to verify such issues automatically soon: https://github.com/nnstreamer/nnstreamer/issues/2638 (nnstreamer-util's ```nnstreamer-check```).
