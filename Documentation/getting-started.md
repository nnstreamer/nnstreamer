## Getting Started

### Prerequisites
The following dependencies are needed to compile/build/run.
* gcc/g++
* gstreamer 1.0 and its relatives
* glib 2.0
* meson >= 0.40

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
$ export DEB_BUILD_OPTIONS="parallel=8"
$ time debuild -us -uc
$ export DEB_BUILD_OPTIONS=""
```
If there is a missing package, debuild will tell you which package is missing.
If you haven't configured debuild properly, yet, you will need to add ```-uc -us``` options to ```debuild```.


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
