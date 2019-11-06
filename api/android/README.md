# NNStreamer API Library for Android

## Prerequisite

We assume that you already have experienced Android application developments with Android Studio.

 * Host PC:
   * OS: Ubuntu 16.04 x86_64 LTS
   * Android Studio: Ubuntu version
   * Android SDK: Min version 24 (Nougat)
   * Android NDK: Use default ndk-bundle in Android Studio
   * GStreamer: gstreamer-1.0-android-universal-1.16.1

## Build library

#### Environment variables

First of all, you need to set-up the development environment as following:

```bash
$ mkdir -p $HOME/android/tools/sdk
$ mkdir -p $HOME/android/gstreamer-1.0
$ mkdir -p $HOME/android/workspace
$
$ vi ~/.bashrc
# Environment variables for developing a NNStreamer application
# $ANDROID_DEV_ROOT/gstreamer-1.0                # GStreamer binaries
# $ANDROID_DEV_ROOT/tools/sdk                    # Android SDK root directory (default location: $HOME/Android/Sdk)
# $ANDROID_DEV_ROOT/workspace/nnstreamer         # NNStreamer cloned git repository
#
export JAVA_HOME=/opt/android-studio/jre            # JRE path in Android Studio
export ANDROID_DEV_ROOT=$HOME/android               # Set your own path (The default path will be "$HOME/Android".)
export ANDROID_SDK=$ANDROID_DEV_ROOT/tools/sdk      # Android SDK (The default path will be "$HOME/Android/Sdk".)
export ANDROID_HOME=$ANDROID_SDK
export GSTREAMER_ROOT_ANDROID=$ANDROID_DEV_ROOT/gstreamer-1.0
export NNSTREAMER_ROOT=$ANDROID_DEV_ROOT/workspace/nnstreamer
```

#### Download Android Studio

Download and install Android Studio to compile an Android source code.
You can see the installation guide [here](https://developer.android.com/studio/install).

For example,
```bash
$ firefox  https://developer.android.com/studio
Then, download "Android Studio" in the /opt folder.
$ cd /opt
$ wget https://dl.google.com/dl/android/studio/ide-zips/3.4.0.18/android-studio-ide-183.5452501-linux.tar.gz
$ tar xvzf ./android-studio-ide-183.5452501-linux.tar.gz
```

#### Download NDK

Use the default NDK in Android Studio.

If you need to set a specific version, download and decompress it to compile normally a GStreamer-based plugin (e.g., NNStreamer).
You can download older version from [here](https://developer.android.com/ndk/downloads/older_releases.html).

#### Download GStreamer binaries

You can get the prebuilt GStreamer binaries from [here](https://gstreamer.freedesktop.org/data/pkg/android/).

For example,
```bash
$ cd $ANDROID_DEV_ROOT/
$ mkdir gstreamer-1.0
$ cd gstreamer-1.0
$ wget https://gstreamer.freedesktop.org/data/pkg/android/1.16.1/gstreamer-1.0-android-universal-1.16.1.tar.xz
$ tar xJf gstreamer-1.0-android-universal-1.16.1.tar.xz
```

Modify the gstreamer-1.0.mk file for NDK build to prevent build error.

```
$GSTREAMER_ROOT_ANDROID/{Target-ABI}/share/gst-android/ndk-build/gstreamer-1.0.mk
```

- Add directory separator.

```
# Add separator '/' between $(GSTREAMER_NDK_BUILD_PATH) and $(plugin)

GSTREAMER_PLUGINS_CLASSES    := $(strip \
            $(subst $(GSTREAMER_NDK_BUILD_PATH),, \
            $(foreach plugin,$(GSTREAMER_PLUGINS), \
            $(wildcard $(GSTREAMER_NDK_BUILD_PATH)/$(plugin)/*.java))))

GSTREAMER_PLUGINS_WITH_CLASSES := $(strip \
            $(subst $(GSTREAMER_NDK_BUILD_PATH),, \
            $(foreach plugin, $(GSTREAMER_PLUGINS), \
            $(wildcard $(GSTREAMER_NDK_BUILD_PATH)/$(plugin)))))

copyjavasource_$(TARGET_ARCH_ABI):
    ...
    $(hide)$(foreach file,$(GSTREAMER_PLUGINS_CLASSES), \
        $(call host-cp,$(GSTREAMER_NDK_BUILD_PATH)/$(file),$(GSTREAMER_JAVA_SRC_DIR)/org/freedesktop/gstreamer/$(file)) && ) echo Done cp
```

#### Download NNStreamer source code

```bash
$ cd $ANDROID_DEV_ROOT/workspace
$ git clone https://github.com/nnsuite/nnstreamer.git
```

#### Build Android API

Run the build script in NNStreamer.
After building the Android API, you can find the library(.aar) in ```$NNSTREAMER_ROOT/android_lib```.

```bash
$ cd $NNSTREAMER_ROOT
$ bash ./api/android/build-android-lib.sh
```

#### Run the unit-test (Optional)

Before running the unit-test, you should download the test model and copy it into your target device manually.

Make directory and copy test model and label files into the internal storage of your own Android target device.

You can download these files from [nnsuite testcases repository](https://github.com/nnsuite/testcases/tree/master/DeepLearningModels/tensorflow-lite/Mobilenet_v1_1.0_224_quant).

```
# You must put the below model and label files in the internal storage of your Android target device.
{INTERNAL_STORAGE}/nnstreamer/test/mobilenet_v1_1.0_224_quant.tflite
{INTERNAL_STORAGE}/nnstreamer/test/add.tflite
{INTERNAL_STORAGE}/nnstreamer/test/labels.txt
{INTERNAL_STORAGE}/nnstreamer/test/orange.png
```

To check the testcases, run the build script with an option ```--run_unittest=yes```.
You can find the result in ```$NNSTREAMER_ROOT/android_lib```.

```bash
$ cd $NNSTREAMER_ROOT
$ bash ./api/android/build-android-lib.sh --run_unittest=yes
```
