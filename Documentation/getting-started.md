## Getting Started

### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++
* gstreamer 1.0 and its relatives
* glib 2.0
* cmake >= 2.8

### Linux Self-Hosted Build

**Approach 1.** Build with debian tools

* How to use mk-build-deps
```bash
$ mk-build-deps --install debian/control
$ dpkg -i nnstreamer-build-deps_2018.6.16_all.deb
```
Note that the version name may change. Please check your local directory after excecuting ```mk-build-deps```.

* How to use debuild
```bash
$ debuild
```
If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add ```-uc -us``` options to ```debuild```.


**Approach 2.** Build with Cmake

At the git repo root directory,
```bash
$ mkdir -p build  # We recommend to build in a "build" directory
$ cd build
$ rm -rf *        # Ensure the build directory is empty
$ cmake ..
$ make
$ cd ..
```

You may copy the resulting plugin (.so file) to gstreamer plugin repository. Or do
```bash
$ cd build
$ sudo make install
```
if installing NNstreamer plugin libraries into ```%{_libdir}```.


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

Then, create tarball that will contain your chroot environment to build package.
```bash
$ vi ~/.pbuilderrc
# man 5 pbuilderrc
DISTRIBUTION=xenial
OTHERMIRROR="deb http://archive.ubuntu.com/ubuntu xenial universe multiverse"
$ sudo ln -s  ~/.pbuilderrc /root/.pbuilderrc
$ sudo pbuilder create
$ ls -al /var/cache/pbuilder/base.tgz
```

Generates .deb packages:
```bash
$ pdebuild
$ ls -al /var/cache/pbuilder/result/*.deb
```
Note that ```pdebuild``` does not execute unit testing.
