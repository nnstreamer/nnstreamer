#!/usr/bin/env bash

##
# @file  build-android-lib.sh
# @brief A script to build NNStreamer API library for Android
#
# Before running this script, below variables must be set.
# - ANDROID_HOME: Android SDK
# - GSTREAMER_ROOT_ANDROID: GStreamer prebuilt libraries for Android
# - NNSTREAMER_ROOT: NNStreamer root directory
#

# Set target ABI (default 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64')
nnstreamer_target_abi="'armeabi-v7a', 'arm64-v8a'"

# Function to check if a package is installed
function check_package() {
    which "$1" 2>/dev/null || {
        echo "Need to install $1."
        exit 1
    }
}

# Check required packages
check_package svn
check_package sed

# Android SDK (Set your own path)
[ -z "$ANDROID_HOME" ] && echo "Need to set ANDROID_HOME." && exit 1

echo "Android SDK: $ANDROID_HOME"

# GStreamer prebuilt libraries for Android
# Download from https://gstreamer.freedesktop.org/data/pkg/android/
[ -z "$GSTREAMER_ROOT_ANDROID" ] && echo "Need to set GSTREAMER_ROOT_ANDROID." && exit 1

echo "GStreamer binaries: $GSTREAMER_ROOT_ANDROID"

# NNStreamer root directory
[ -z "$NNSTREAMER_ROOT" ] && echo "Need to set NNSTREAMER_ROOT." && exit 1

echo "NNStreamer root directory: $NNSTREAMER_ROOT"

echo "Start to build NNStreamer library for Android."

# Modify header for Android
cd $NNSTREAMER_ROOT/api/capi
./modify_nnstreamer_h_for_nontizen.sh
cd $NNSTREAMER_ROOT

# Make directory to build NNStreamer library
mkdir -p build_android_lib

# Copy the files (native and java to build Android library) to build directory
cp -r $NNSTREAMER_ROOT/api/android/* ./build_android_lib

# Get the prebuilt libraries and build-script
svn --force export https://github.com/nnsuite/nnstreamer-android-resource/trunk/android_api ./build_android_lib

pushd ./build_android_lib

tar xJf ./ext-files/ext-files.tar.xz -C ./api
tar xJf ./ext-files/tensorflow-lite_arm64.tar.xz -C ./api/jni
tar xJf ./ext-files/tensorflow-lite_armv7.tar.xz -C ./api/jni
tar xJf ./ext-files/tensorflow-lite_x86.tar.xz -C ./api/jni
tar xJf ./ext-files/tensorflow-lite_x86_64.tar.xz -C ./api/jni

sed -i "s|abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'|abiFilters $nnstreamer_target_abi|" api/build.gradle

echo "Starting gradle build for Android library."
./gradlew api:assembleRelease

popd

# Check if build procedure is done.
nnstreamer_android_api_lib=./build_android_lib/api/build/outputs/aar/api-release.aar
result_directory=android_lib

if [[ -e $nnstreamer_android_api_lib ]]; then
    echo "Build procedure is done, copy NNStreamer library to $result_directory directory."
    mkdir -p $result_directory
    cp $nnstreamer_android_api_lib ./$result_directory/nnstreamer-api-$(date '+%Y-%m-%d').aar
else
    echo "Failed to build NNStreamer library."
fi

# Remove build directory
rm -rf build_android_lib

