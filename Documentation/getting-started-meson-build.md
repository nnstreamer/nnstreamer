---
title: Meson/Ninja Build
...

## Getting Started: Meson/Ninja Build

NNStreamer uses meson/ninja as its standard build environment; both Tizen(GBS-OBS/.rpm) and Ubuntu(pdebuild/.deb) use meson/ninja.

With meson/ninja, you may build in the git repo directory and install nnstreamer binaries to your designated directories.
However, you should be careful on configuring nnstreamer; i.e., the paths of subplugins and .ini file locations.
Please note that nnstreamer tries to load /etc/nnstreamer.ini by default; step 7 below describes how the ini file is chosen.

* https://mesonbuild.com/Getting-meson.html

This document assumes that you are using Ubuntu.


### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++ (C++17 for C++-class filters)
* gstreamer 1.0 and its relatives
* glib 2.0
* meson >= 0.57.0
* ninja-build
* Neural network frameworks or libraries for plugins (e.g., tensorflow) you want to use, including their pkgconfig files or mechanisms to allow meson to discover its headers and libraries. If you use development packages packaged by us for tensorflow/pytorch/openvino/..., you do not need to worry.
    * Possible frameworks for "extra" plugins: tensorflow, tensorflow-lite, pytorch, protobuf, flatbuf, openvino, ncsdk2, Verisilicon-Vivante, SNPE, TensorRT, mqtt, ...
* [SSAT](https://github.com/myungjoo/SSAT) (optional. required for unit testing)
* gtest (optional. required for unit testing)

The minimal requirement to build nnstreamer with default configuration is
```bash
$ sudo apt-get install meson ninja-build gcc g++ libglib2.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libjson-glib-dev
```

Optional utilities:
```bash
$ sudo add-apt-repository ppa:nnstreamer/ppa
$ sudo apt-get update
$ sudo apt-get install libiniparser-dev=4.1 # if your iniparser is < 4.1
$ sudo apt-get install ssat # if you want to run unit tests
```
You may build and install iniparser and ssat directly from their sources, too.

***1. Clone nnstreamer***

```bash
$ git clone https://github.com/nnstreamer/nnstreamer
```

***2. Install related frameworks***


*Option A*: You may install optional developmental packages from nnstreamer PPA for more subplugins and extra plugins (choose what you need):
```
$ sudo add-apt-repository ppa:nnstreamer/ppa
$ sudo apt-get update
$ sudo apt-get install libedgetpu-dev libflatbuffers-dev libgrpc-dev openvino-dev libpaho-mqtt-dev libprotobuf-dev pytorch tensorflow2-lite-dev tvm-runtime-dev
```

*Option B*: Build & install your own frameworks and provide corresponding pkg-config files.
[tensorflow2-lite pkgconfig file template](https://git.tizen.org/cgit/platform/upstream/tensorflow2/tree/packaging/tensorflow2-lite.pc.in?h=tizen) /
[tensorflow1-lite pkgconfig file template](https://git.tizen.org/cgit/platform/upstream/tensorflow/tree/packaging/tensorflow-lite.pc.in?h=tizen) /
[tensorflow1 pkgconfig file template](https://git.tizen.org/cgit/platform/upstream/tensorflow/tree/packaging/tensorflow-lite.pc.in?h=tizen)

Fill in paths of libdir and includedir of the frameworks and place the .pc files at /usr/lib/pkgconfig/

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
Or You can set your own path to install libraries and header files.
```bash
# Configure and build example.
$ sudo vi ~/.bashrc

export NNST_ROOT=$HOME/nnstreamer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NNST_ROOT/lib
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$NNST_ROOT/lib/gstreamer-1.0
# Include NNStreamer headers and libraries
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$NNST_ROOT/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NNST_ROOT/include
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$NNST_ROOT/lib/pkgconfig

$ source ~/.bashrc

$ meson --prefix=${NNST_ROOT} --sysconfdir=${NNST_ROOT} --libdir=lib --bindir=bin --includedir=include build
$ ninja -C build install
```

***4. Build***

Assuming you have configured meson at ```./build```.
```bash
$ ninja -C build
```


***5. Install***

Assuming you have configured meson at ```./build```.
```bash
$ sudo ninja -C build install  # if the destination is not user directory
```

Or if you want somewhere else:
```bash
$ DESTDIR=/home/me/somewhereelse/ ninja -C build install
```

Then, it will install:
- nnstreamer plugins to ```{libdir}/gstreamer-1.0/```
- libraries to ```{libdir}/```
- subplugins to ```{libdir}/nnstreamer/PLUGIN-TYPE/``` (or ```{subplugindir}/PLUGIN-TYPE/``` when ```-Dsubplugindir``` is set)
- common header files to ```{includedir}/```

The python3 sub-plugins provide the ```nnstreamer_python``` module to the scripts they run. To import it from a python program of your own, put a link named ```nnstreamer_python.so``` to ```{libdir}/nnstreamer_python3.so``` in a directory on ```PYTHONPATH```, as the Debian and Tizen packages do in their python3 site directory.


***6. Verifying installation***

Assuming you have configured meson at ```./build```.
```bash
$ build/tools/development/confchk/nnstreamer-check

...

```

```nnstreamer-check``` utility shows the status of your nnstreamer installation.
It uses environmental-variables, nnstreamer.ini file, and hardcoded default values, with the respective order.

***7. Configuration file and the environment***

nnstreamer reads one main ini file, chosen in this order:
1. the file named by the ```NNSTREAMER_CONF``` environment variable, if it is a regular file (not on Tizen),
2. ```{sysconfdir}/nnstreamer.ini``` (under ```{prefix}``` when ```sysconfdir``` is relative),
3. ```/etc/nnstreamer.ini```,
4. the file named by ```NNSTREAMER_CONF```, when neither of the two above exists (Tizen included).

The chosen ini sets the subplugin directories, ```enable_symlink```, ```enable_envvar``` and ```extra_config_path```, an extra ini whose subplugin directories are added as well.
```enable_envvar``` (false when the key is absent) decides whether ```NNSTREAMER_FILTERS```, ```NNSTREAMER_DECODERS```, ```NNSTREAMER_CUSTOMFILTERS```, ```NNSTREAMER_CONVERTERS```, ```NNSTREAMER_TRAINERS``` and the ```NNSTREAMER_<group>_<key>``` overrides of ini values, spelled case-sensitively as in the ini (e.g., ```NNSTREAMER_filter_framework_priority_tflite``` for the ```framework=auto``` priority), are read; ```-Denable-env-var=false``` writes ```enable_envvar=false``` into the installed ini and does not disable ```NNSTREAMER_CONF```.

nnstreamer trusts the environment of its process, as GStreamer does with ```GST_PLUGIN_PATH```: these variables select the shared objects nnstreamer loads, and nnstreamer does not ignore them in setuid, setgid or file-capability programs.
Do not run nnstreamer in such a program with an environment you do not control; clear ```NNSTREAMER_*``` and ```GST_*``` before starting it, and keep the ini files and the subplugin directories writable only by root or another trusted account (e.g., in a read-only partition).
