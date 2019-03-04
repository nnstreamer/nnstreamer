# How to build C/C++ source code with nkd-build 

This manual is to describe how to generate the .so files from native C/C++
source code. In general, these .so files are used as a low-level library
for Android Application framework. Then, These files have been enabled by
Android application via JNI interface.
The libnnstreamer.so file is to be used as a native libraries on Android devices.
 * Host PC: Ubuntu 16.04 x86_64 LTS
 * CPU Architecture: ARM 64bit (aarch64)
 * Android platform: 7.0 (Nougat)
 * Android NDK: r12b
 * Android API level: 24

## Set-up Android NDK
```bash
cd ~/android/
wget https://dl.google.com/android/repository/android-ndk-r12b-linux-x86_64.zip
vi ~/.bashrc
export ANDROID_NDK=~/android/android-ndk-r12b
export PATH=$ANDROID_NDK:$PATH
```

## Download prebuilt Android-Gstreamer libraries
Please download required files such as "*.tar.bz2" from http://nnsuite.mooo.com/warehouse/.
 * gstreamer-prebuilts-for-android-device/gst_root_android-custom-1.12.4-ndkr12b-20190213-0900/
```bash
vi ~/.bashrc
export GSTREAMER_ROOT_ANDROID=~/android/gst_root_android
mkdir -p ~/android/gst_root_android/arm64
```

## How to build a NNstreamer library
```bash
cd ./jni
# We recommend that you always remove the libs and obj folder to avoid an unexpected binary inconsistency.
rm -rf ../libs/ ../obj/
ndk-build NDK_PROJECT_PATH=.  APP_BUILD_SCRIPT=./Android-nnstreamer.mk NDK_APPLICATION_MK=./Application.mk -j$(nproc)
ls -al ../libs/arm64-v8a/libnnstreamer.so
```

## How to build a test application
```bash
cd ./jni
ndk-build NDK_PROJECT_PATH=.  APP_BUILD_SCRIPT=./Android-app.mk NDK_APPLICATION_MK=./Application.mk -j$(nproc)
ls -al ../libs/arm64-v8a/
```
