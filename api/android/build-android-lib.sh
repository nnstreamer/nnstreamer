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
# To include sub-plugin for SNAP, you also should define the variable 'SNAP_DIRECTORY'.
# - SNAP_DIRECTORY: Absolute path for SNAP, tensor-filter sub-plugin and prebuilt library.
#
# Build options
# --api_option (default 'all', 'lite' to build with GStreamer core plugins)
# --target_abi (default 'armv7, arm64')
# --run_test (default 'no', 'yes' to run the instrumentation test)
# --enable_snap (default 'no', 'yes' to build with sub-plugin for SNAP)
#
# For example, to build library with core plugins for arm64-v8a
# ./build-android-lib.sh --api_option=lite --target_abi=arm64
#

# API build option ('lite' to build with GStreamer core plugins)
nnstreamer_api_option=all

# Set target ABI ('armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64')
nnstreamer_target_abi="'armeabi-v7a', 'arm64-v8a'"

# Set tensorflow-lite version (available: 1.9 and 1.13)
nnstreamer_tf_lite_ver=1.13

# Run instrumentation test after build procedure is done
run_test='no'

# Variables to release library (GROUP:ARTIFACT:VERSION)
release_bintray='no'

# Enable SNAP
enable_snap='no'

# Parse args
for arg in "$@"; do
    case $arg in
        --api_option=*)
            nnstreamer_api_option=${arg#*=}
            ;;
        --target_abi=*)
            target_abi=${arg#*=}
            if [[ $target_abi == 'armv7' ]]; then
                nnstreamer_target_abi="'armeabi-v7a'"
            elif [[ $target_abi == 'arm64' ]]; then
                nnstreamer_target_abi="'arm64-v8a'"
            else
                echo "Unknown target ABI." && exit 1
            fi
            ;;
        --release=*)
            release_bintray=${arg#*=}
            ;;
        --release_version=*)
            release_version=${arg#*=}
            ;;
        --bintray_user_name=*)
            bintray_user_name=${arg#*=}
            ;;
        --bintray_user_key=*)
            bintray_user_key=${arg#*=}
            ;;
        --run_test=*)
            run_test=${arg#*=}
            ;;
        --result_directory=*)
            result_directory=${arg#*=}
            ;;
        --enable_snap=*)
            enable_snap=${arg#*=}
            ;;
    esac
done

if [[ $enable_snap == 'yes' ]]; then
    [ -z "$SNAP_DIRECTORY" ] && echo "Need to set SNAP_DIRECTORY, to build sub-plugin for SNAP." && exit 1

    echo "Set target ABI arm64-v8a, including SNAP: $SNAP_DIRECTORY"
    target_abi='arm64'
    nnstreamer_target_abi="'arm64-v8a'"
fi

if [[ $release_bintray == 'yes' ]]; then
    [ -z "$release_version" ] && echo "Set release version." && exit 1
    [ -z "$bintray_user_name" ] || [ -z "$bintray_user_key" ] && echo "Set user info to release." && exit 1

    echo "Release version: $release_version user: $bintray_user_name"
fi

# Set library name
nnstreamer_lib_name="nnstreamer"

if [[ $nnstreamer_api_option != 'all' ]]; then
    nnstreamer_lib_name="$nnstreamer_lib_name-$nnstreamer_api_option"
fi

if [[ -n $target_abi ]]; then
    nnstreamer_lib_name="$nnstreamer_lib_name-$target_abi"
fi

echo "NNStreamer library name: $nnstreamer_lib_name"

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
check_package zip

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
pushd $NNSTREAMER_ROOT

# Make directory to build NNStreamer library
mkdir -p build_android_lib

# Copy the files (native and java to build Android library) to build directory
cp -r ./api/android/* ./build_android_lib

# Get the prebuilt libraries and build-script
svn --force export https://github.com/nnsuite/nnstreamer-android-resource/trunk/android_api ./build_android_lib

pushd ./build_android_lib

tar xJf ./ext-files/tensorflow-lite-$nnstreamer_tf_lite_ver.tar.xz -C ./api/src/main/jni

# Update target ABI
sed -i "s|abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'|abiFilters $nnstreamer_target_abi|" api/build.gradle

# Update API build option
sed -i "s|NNSTREAMER_API_OPTION := all|NNSTREAMER_API_OPTION := $nnstreamer_api_option|" api/src/main/jni/Android.mk

# Update SNAP option
if [[ $enable_snap == 'yes' ]]; then
    sed -i "s|ENABLE_SNAP := false|ENABLE_SNAP := true|" ext-files/jni/Android-nnstreamer-prebuilt.mk
    sed -i "s|ENABLE_SNAP := false|ENABLE_SNAP := true|" api/src/main/jni/Android.mk
    cp -r $SNAP_DIRECTORY/* api/src/main/jni
fi

# Add dependency for release
if [[ $release_bintray == 'yes' ]]; then
    sed -i "s|// add dependency (bintray)|classpath 'com.novoda:bintray-release:0.9.1'|" build.gradle

    sed -i "s|// add plugin (bintray)|apply plugin: 'com.novoda.bintray-release'\n\
\n\
publish {\n\
    userOrg = 'nnsuite'\n\
    repoName = 'nnstreamer'\n\
    groupId = 'org.nnsuite'\n\
    artifactId = '$nnstreamer_lib_name'\n\
    publishVersion = '$release_version'\n\
    desc = 'NNStreamer API for Android'\n\
    website = 'https://github.com/nnsuite/nnstreamer'\n\
    issueTracker = 'https://github.com/nnsuite/nnstreamer/issues'\n\
    repository = 'https://github.com/nnsuite/nnstreamer.git'\n\
}|" api/build.gradle
fi

echo "Starting gradle build for Android library."

chmod +x gradlew
# Build Android library.
./gradlew api:build

# Check if build procedure is done.
nnstreamer_android_api_lib=./api/build/outputs/aar/api-release.aar

result=1
if [[ -e $nnstreamer_android_api_lib ]]; then
    if [[ -z $result_directory ]]; then
        result_directory=../android_lib
    fi
    today=$(date '+%Y-%m-%d')
    result=0

    echo "Build procedure is done, copy NNStreamer library to $result_directory directory."
    mkdir -p $result_directory
    cp $nnstreamer_android_api_lib $result_directory/$nnstreamer_lib_name-$today.aar

    # Prepare native libraries and header files for C-API
    unzip $nnstreamer_android_api_lib -d aar_extracted

    mkdir -p main/assets
    mkdir -p main/java/org/freedesktop
    mkdir -p main/jni/nnstreamer/lib
    mkdir -p main/jni/nnstreamer/include

    cp -r api/src/main/java/org/freedesktop/* main/java/org/freedesktop
    cp -r aar_extracted/assets/* main/assets
    cp -r aar_extracted/jni/* main/jni/nnstreamer/lib
    cp ext-files/jni/Android-nnstreamer-prebuilt.mk main/jni
    # header for C-API
    cp $NNSTREAMER_ROOT/api/capi/include/nnstreamer.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/api/capi/include/nnstreamer-single.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/api/capi/include/platform/tizen_error.h main/jni/nnstreamer/include
    # header for plugin
    cp $NNSTREAMER_ROOT/gst/nnstreamer/nnstreamer_plugin_api.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/gst/nnstreamer/nnstreamer_plugin_api_converter.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/gst/nnstreamer/nnstreamer_plugin_api_decoder.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/gst/nnstreamer/nnstreamer_plugin_api_filter.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/gst/nnstreamer/tensor_filter_custom.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/gst/nnstreamer/tensor_filter_custom_easy.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/gst/nnstreamer/tensor_typedef.h main/jni/nnstreamer/include
    cp $NNSTREAMER_ROOT/ext/nnstreamer/tensor_filter/tensor_filter_cpp.h main/jni/nnstreamer/include

    nnstreamer_native_files="$nnstreamer_lib_name-native-$today.zip"
    zip -r $nnstreamer_native_files main
    cp $nnstreamer_native_files $result_directory

    rm -rf aar_extracted main

    # Upload to jcenter
    if [[ $release_bintray == 'yes' ]]; then
        echo "Upload NNStreamer library to Bintray."
        ./gradlew api:bintrayUpload -PbintrayUser=$bintray_user_name -PbintrayKey=$bintray_user_key -PdryRun=false
    fi

    # Run instrumentation test
    if [[ $run_test == 'yes' ]]; then
        echo "Run instrumentation test."
        ./gradlew api:connectedCheck

        test_result="$nnstreamer_lib_name-test-$today.zip"
        zip -r $test_result api/build/reports
        cp $test_result $result_directory
    fi
else
    echo "Failed to build NNStreamer library."
fi

popd

# Remove build directory
rm -rf build_android_lib

popd

# exit with success/failure status
exit $result
