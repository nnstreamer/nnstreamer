#!/usr/bin/env bash

##
## SPDX-License-Identifier: LGPL-2.1-only
##
# @file  build-android-lib.sh
# @brief A script to build NNStreamer API library for Android
#
# The following comments that start with '##@@' are for the generation of usage messages.
##@@ Build script for Android NNStreamer API Library
##@@  - Before running this script, below variables must be set.
##@@  - ANDROID_SDK_ROOT: Android SDK
##@@  - ANDROID_NDK_ROOT: Android NDK
##@@  - GSTREAMER_ROOT_ANDROID: GStreamer prebuilt libraries for Android
##@@  - NNSTREAMER_ROOT: The source root directory of NNStreamer
##@@
##@@ usage: build-android-lib.sh [OPTIONS]
##@@
##@@ basic options:
##@@   --help
##@@       display this help and exit
##@@   --build-type=(all|lite|single|internal)
##@@       'all'      : default
##@@       'lite'     : build with GStreamer core plugins
##@@       'single'   : no plugins, single-shot only
##@@       'internal' : no plugins except for enable single-shot only, enable NNFW only
##@@   --target_abi=(armeabi-v7a|arm64-v8a)
##@@       'arm64-v8a' is the default Android ABI
##@@   --run_test=(yes|no)
##@@       'yes'      : run instrumentation test after build procedure is done
##@@       'no'       : [default]
##@@   --nnstreamer_dir=(the_source_root_of_nnstreamer)
##@@       This option overrides the NNSTREAMER_ROOT variable
##@@
##@@ options for tensor filter sub-plugins:
##@@   --enable_snap=(yes|no)
##@@       'yes'      : build with sub-plugin for SNAP
##@@                    This option requires 1n additional variable, 'SNAP_DIRECTORY',
##@@                    which indicates the SNAP SDK interface's absolute path.
##@@       'no'       : [default]
##@@   --enable_nnfw=(yes|no)
##@@       'yes'      : [default]
##@@       'no'       : build without the sub-plugin for NNFW
##@@   --enable_snpe=(yes|no)
##@@       'yes'      : build with sub-plugin for SNPE
##@@       'no'       : [default]
##@@   --enable_pytorch=(yes(:(1.8.0))?|no)
##@@       'yes'      : build with sub-plugin for PyTorch. You can optionally specify the version of
##@@                    PyTorch to use by appending ':version' [1.8.0 is the default].
##@@       'no'       : [default] build without the sub-plugin for PyTorch
##@@   --enable_tflite=(yes(:(1.9|1.13.1|1.15.2|2.3.0))?|no)
##@@       'yes'      : [default] you can optionally specify the version of tensorflow-lite to use
##@@                    by appending ':version' [1.13.1 is the default].
##@@       'no'       : build without the sub-plugin for tensorflow-lite
##@@
##@@ options for tensor decoder sub-plugins:
##@@   --enable_decoder_flatbuf=(yes|no)
##@@       'yes'      : [default]
##@@       'no'       : build without the sub-plugin for FlatBuffers
##@@
##@@ For example, to build library with core plugins for arm64-v8a
##@@  ./build-android-lib.sh --api_option=lite --target_abi=arm64-v8a

set -e

# API build option
# 'all' : default
# 'lite' : with GStreamer core plugins
# 'single' : no plugins, single-shot only
# 'internal' : no plugins, single-shot only, enable NNFW only
build_type="all"

nnstreamer_api_option="all"
include_assets="no"

# Set target ABI ('armeabi-v7a', 'arm64-v8a')
target_abi="arm64-v8a"

# Run instrumentation test after build procedure is done
run_test="no"

# Variables to release library (GROUP:ARTIFACT:VERSION)
release_bintray="no"

# Enable SNAP
enable_snap="no"

# Enable NNFW
enable_nnfw="yes"

# Enable SNPE
enable_snpe="no"

# Enable PyTorch
enable_pytorch="no"

# Set PyTorch version (available: 1.8.0 (unstable))
pytorch_ver="1.8.0"
pytorch_vers_support="1.8.0"

# Enable tensorflow-lite
enable_tflite="yes"

# Enable the flatbuffer decoder by default
enable_decoder_flatbuf="yes"
decoder_flatbuf_ver="1.12.0"

# Set tensorflow-lite version (available: 1.9.0 / 1.13.1 / 1.15.2 / 2.3.0)
tf_lite_ver="1.13.1"
tf_lite_vers_support="1.9.0 1.13.1 1.15.2 2.3.0"

# Set NNFW version (https://github.com/Samsung/ONE/releases)
nnfw_ver="1.12.0"
enable_nnfw_ext="no"

# Find '--help' in the given arguments
arg_help="--help"
for arg in "$@"; do
    if [[ $arg == $arg_help ]]; then
        sed -ne 's/^##@@ \(.*\)/\1/p' $0 && exit 1
    fi
done

# Parse args
for arg in "$@"; do
    case $arg in
        --build_type=*)
            build_type=${arg#*=}
            ;;
        --target_abi=*)
            target_abi=${arg#*=}
            if [ $target_abi != "armeabi-v7a" ] && [ $target_abi != "arm64-v8a" ]; then
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
        --nnstreamer_dir=*)
            nnstreamer_dir=${arg#*=}
            ;;
        --result_dir=*)
            result_dir=${arg#*=}
            ;;
        --gstreamer_dir=*)
            gstreamer_dir=${arg#*=}
            ;;
        --android_sdk_dir=*)
            android_sdk_dir=${arg#*=}
            ;;
        --android_ndk_dir=*)
            android_ndk_dir=${arg#*=}
            ;;
        --enable_snap=*)
            enable_snap=${arg#*=}
            ;;
        --enable_nnfw=*)
            enable_nnfw=${arg#*=}
            ;;
        --enable_nnfw_ext=*)
            enable_nnfw_ext=${arg#*=}
            ;;
        --enable_snpe=*)
            enable_snpe=${arg#*=}
            ;;
        --enable_pytorch=*)
            IFS=':' read -ra enable_pytorch_args <<< "${arg#*=}"
            is_valid_pytorch_version=0
            enable_pytorch=${enable_pytorch_args[0]}
            if [[ ${enable_pytorch} == "yes" ]]; then
                if [[ ${enable_pytorch_args[1]} == "" ]]; then
                    break
                fi
                for ver in ${pytorch_vers_support}; do
                    if [[ ${ver} == ${enable_pytorch_args[1]} ]]; then
                        is_valid_pytorch_version=1
                        pytorch_ver=${ver}
                        break
                    fi
                done
                if [[ ${is_valid_pytorch_version} == 0 ]]; then
                    printf "'%s' is not a supported version of PyTorch." "${enable_pytorch_args[1]}"
                    printf "The default version, '%s', will be used.\n"  "${pytorch_ver}"
                fi
            fi
            ;;
        --enable_tflite=*)
            IFS=':' read -ra enable_tflite_args <<< "${arg#*=}"
            is_valid_tflite_version=0
            enable_tflite=${enable_tflite_args[0]}
            if [[ ${enable_tflite} == "yes" ]]; then
                if [[ ${enable_tflite_args[1]} == "" ]]; then
                    break
                fi
                for ver in ${tf_lite_vers_support}; do
                    if [[ ${ver} == ${enable_tflite_args[1]} ]]; then
                        is_valid_tflite_version=1
                        tf_lite_ver=${ver}
                        break
                    fi
                done
                if [[ ${is_valid_tflite_version} == 0 ]]; then
                    printf "'%s' is not a supported version of TensorFlow Lite." "${enable_tflite_args[1]}"
                    printf "The default version, '%s', will be used.\n"  "${tf_lite_ver}"
                fi
            fi
            ;;
        --enable_decoder_flatbuf=*)
            enable_decoder_flatbuf=${arg#*=}
            ;;
    esac
done

# Check build type
if [[ $build_type == "single" ]]; then
    nnstreamer_api_option="single"
elif [[ $build_type == "lite" ]]; then
    nnstreamer_api_option="lite"
elif [[ $build_type == "internal" ]]; then
    nnstreamer_api_option="single"

    enable_snap="no"
    enable_nnfw="yes"
    enable_nnfw_ext="yes"
    enable_snpe="no"
    enable_pytorch="no"
    enable_tflite="no"

    target_abi="arm64-v8a"
elif [[ $build_type != "all" ]]; then
    echo "Failed, unknown build type $build_type." && exit 1
fi

if [[ $enable_snap == "yes" ]]; then
    [ -z "$SNAP_DIRECTORY" ] && echo "Need to set SNAP_DIRECTORY, to build sub-plugin for SNAP." && exit 1
    [ $target_abi != "arm64-v8a" ] && echo "Set target ABI arm64-v8a to build sub-plugin for SNAP." && exit 1

    echo "Build with SNAP: $SNAP_DIRECTORY"
fi

if [[ $enable_nnfw == "yes" ]]; then
    [ $target_abi != "arm64-v8a" ] && echo "Set target ABI arm64-v8a to build sub-plugin for NNFW." && exit 1

    echo "Build with NNFW $nnfw_ver"

    if [[ $enable_nnfw_ext == "yes" ]]; then
        [ -z "$NNFW_DIRECTORY" ] && echo "Need to set NNFW_DIRECTORY, to get NNFW-ext library." && exit 1
    fi
fi

if [[ $enable_snpe == "yes" ]]; then
    [ $enable_snap == "yes" ] && echo "DO NOT enable SNAP and SNPE both. The app would fail to use DSP or NPU runtime." && exit 1
    [ -z "$SNPE_DIRECTORY" ] && echo "Need to set SNPE_DIRECTORY, to build sub-plugin for SNPE." && exit 1
    [ $target_abi != "arm64-v8a" ] && echo "Set target ABI arm64-v8a to build sub-plugin for SNPE." && exit 1

    echo "Build with SNPE: $SNPE_DIRECTORY"
fi

if [[ $enable_pytorch == "yes" ]]; then
    echo "Build with PyTorch $pytorch_ver"
fi

if [[ $enable_tflite == "yes" ]]; then
    echo "Build with tensorflow-lite $tf_lite_ver"
fi

if [[ $release_bintray == "yes" ]]; then
    [ -z "$release_version" ] && echo "Set release version." && exit 1
    [ -z "$bintray_user_name" ] || [ -z "$bintray_user_key" ] && echo "Set user info to release." && exit 1

    echo "Release version: $release_version user: $bintray_user_name"
fi

if [[ $enable_decoder_flatbuf == "yes" ]]; then
    echo "Build with flatbuffers v$decoder_flatbuf_ver for the decoder sub-plugin"
fi

# Set library name
nnstreamer_lib_name="nnstreamer"

if [[ $build_type != "all" ]]; then
    nnstreamer_lib_name="$nnstreamer_lib_name-$build_type"
fi

echo "NNStreamer library name: $nnstreamer_lib_name"

# Android SDK (Set your own path)
if [[ -z "$android_sdk_dir" ]]; then
    [ -z "$ANDROID_SDK_ROOT" ] && echo "Need to set ANDROID_SDK_ROOT." && exit 1
    android_sdk_dir=$ANDROID_SDK_ROOT
fi

if [[ -z "$android_ndk_dir" ]]; then
    [ -z "$ANDROID_NDK_ROOT" ] && echo "Need to set ANDROID_NDK_ROOT." && exit 1
    android_ndk_dir=$ANDROID_NDK_ROOT
fi

echo "Android SDK: $android_sdk_dir"
echo "Android NDK: $android_ndk_dir"

echo "Patching NDK source"
# See: https://github.com/nnstreamer/nnstreamer/issues/2899

sed -z -i "s|struct AMediaCodecOnAsyncNotifyCallback {\n      AMediaCodecOnAsyncInputAvailable  onAsyncInputAvailable;\n      AMediaCodecOnAsyncOutputAvailable onAsyncOutputAvailable;\n      AMediaCodecOnAsyncFormatChanged   onAsyncFormatChanged;\n      AMediaCodecOnAsyncError           onAsyncError;\n};|typedef struct AMediaCodecOnAsyncNotifyCallback {\n      AMediaCodecOnAsyncInputAvailable  onAsyncInputAvailable;\n      AMediaCodecOnAsyncOutputAvailable onAsyncOutputAvailable;\n      AMediaCodecOnAsyncFormatChanged   onAsyncFormatChanged;\n      AMediaCodecOnAsyncError           onAsyncError;\n} AMediaCodecOnAsyncNotifyCallback;|" $android_ndk_dir/toolchains/llvm/prebuilt/*/sysroot/usr/include/media/NdkMediaCodec.h

# GStreamer prebuilt libraries for Android
# Download from https://gstreamer.freedesktop.org/data/pkg/android/
if [[ -z "$gstreamer_dir" ]]; then
    [ -z "$GSTREAMER_ROOT_ANDROID" ] && echo "Need to set GSTREAMER_ROOT_ANDROID." && exit 1
    gstreamer_dir=$GSTREAMER_ROOT_ANDROID
fi

echo "GStreamer binaries: $gstreamer_dir"

# NNStreamer root directory
if [[ -z "$nnstreamer_dir" ]]; then
    [ -z "$NNSTREAMER_ROOT" ] && echo "Need to set NNSTREAMER_ROOT." && exit 1
    nnstreamer_dir=$NNSTREAMER_ROOT
fi

echo "NNStreamer root directory: $nnstreamer_dir"

# Build result directory
if [[ -z "$result_dir" ]]; then
    result_dir=$nnstreamer_dir/android_lib
fi

mkdir -p $result_dir

echo "Start to build NNStreamer library for Android (build $build_type)"
pushd $nnstreamer_dir

# Make directory to build NNStreamer library
build_dir=build_android_$build_type
mkdir -p $build_dir

# Copy the files (native and java to build Android library) to build directory
cp -r ./api/android/* ./$build_dir

# Get the prebuilt libraries and build-script
mkdir -p $build_dir/external

svn --force export https://github.com/nnstreamer/nnstreamer-android-resource/trunk/android_api ./$build_dir

# @todo We need another mechanism for downloading third-party/external softwares
rm -f ./$build_dir/external/*.tar.gz ./$build_dir/external/*.tar.xz
if [[ $enable_tflite == "yes" ]]; then
    wget --directory-prefix=./$build_dir/external https://raw.githubusercontent.com/nnstreamer/nnstreamer-android-resource/master/external/tensorflow-lite-$tf_lite_ver.tar.xz
fi

if [[ $enable_nnfw == "yes" ]]; then
    wget --directory-prefix=./$build_dir/external https://github.com/Samsung/ONE/releases/download/$nnfw_ver/onert-$nnfw_ver-android-aarch64.tar.gz
    wget --directory-prefix=./$build_dir/external https://github.com/Samsung/ONE/releases/download/$nnfw_ver/onert-devel-$nnfw_ver.tar.gz

    # You should get ONE-EXT release and copy it into NNFW_DIRECTORY.
    if [[ $enable_nnfw_ext == "yes" ]]; then
        cp $NNFW_DIRECTORY/onert-ext-$nnfw_ver-android-aarch64.tar.gz ./$build_dir/external
    fi
fi

if [[ $enable_pytorch == "yes" ]]; then
    wget --directory-prefix=./$build_dir/external https://raw.githubusercontent.com/nnstreamer/nnstreamer-android-resource/master/external/pytorch-$pytorch_ver.tar.xz
fi

pushd ./$build_dir

# Update target ABI
sed -i "s|abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'|abiFilters '$target_abi'|" api/build.gradle

# Update API build option
sed -i "s|NNSTREAMER_API_OPTION := all|NNSTREAMER_API_OPTION := $nnstreamer_api_option|" api/src/main/jni/Android.mk

if [[ $include_assets == "yes" ]]; then
    sed -i "s|GSTREAMER_INCLUDE_FONTS := no|GSTREAMER_INCLUDE_FONTS := yes|" api/src/main/jni/Android.mk
    sed -i "s|GSTREAMER_INCLUDE_CA_CERTIFICATES := no|GSTREAMER_INCLUDE_CA_CERTIFICATES := yes|" api/src/main/jni/Android.mk
fi

# Update NNStreamer, GStreamer and Android SDK
sed -i "s|nnstreamerRoot=nnstreamer-path|nnstreamerRoot=$nnstreamer_dir|" gradle.properties
sed -i "s|gstAndroidRoot=gstreamer-path|gstAndroidRoot=$gstreamer_dir|" gradle.properties
sed -i "s|ndk.dir=ndk-path|ndk.dir=$android_ndk_dir|" local.properties
sed -i "s|sdk.dir=sdk-path|sdk.dir=$android_sdk_dir|" local.properties

# Update SNAP option
if [[ $enable_snap == "yes" ]]; then
    sed -i "s|ENABLE_SNAP := false|ENABLE_SNAP := true|" api/src/main/jni/Android-nnstreamer-prebuilt.mk
    sed -i "s|ENABLE_SNAP := false|ENABLE_SNAP := true|" api/src/main/jni/Android.mk

    mkdir -p api/src/main/jni/snap/include api/src/main/jni/snap/lib
    cp $SNAP_DIRECTORY/*.h api/src/main/jni/snap/include
    cp $SNAP_DIRECTORY/*.so api/src/main/jni/snap/lib
fi

# Update NNFW option
if [[ $enable_nnfw == "yes" ]]; then
    sed -i "s|ENABLE_NNFW := false|ENABLE_NNFW := true|" api/src/main/jni/Android-nnstreamer-prebuilt.mk
    sed -i "s|ENABLE_NNFW := false|ENABLE_NNFW := true|" api/src/main/jni/Android.mk
    sed -i "$ a NNFW_EXT_LIBRARY_PATH=src/main/jni/nnfw/ext" gradle.properties

    mkdir -p external/nnfw
    tar -zxf external/onert-$nnfw_ver-android-aarch64.tar.gz -C external/nnfw
    tar -zxf external/onert-devel-$nnfw_ver.tar.gz -C external/nnfw

    if [[ $enable_nnfw_ext == "yes" ]]; then
        tar -zxf external/onert-ext-$nnfw_ver-android-aarch64.tar.gz -C external/nnfw
    fi

    # Remove duplicated, unnecessary files (c++shared and tensorflow-lite)
    rm -f external/nnfw/lib/libc++_shared.so
    rm -f external/nnfw/lib/libtensorflowlite_jni.so
    rm -f external/nnfw/lib/libneuralnetworks.so

    mkdir -p api/src/main/jni/nnfw/include api/src/main/jni/nnfw/lib
    mkdir -p api/src/main/jni/nnfw/ext/arm64-v8a
    mv external/nnfw/include/* api/src/main/jni/nnfw/include
    mv external/nnfw/lib/libnnfw-dev.so api/src/main/jni/nnfw/lib
    mv external/nnfw/lib/* api/src/main/jni/nnfw/ext/arm64-v8a
fi

# Update SNPE option
if [[ $enable_snpe == "yes" ]]; then
    sed -i "s|ENABLE_SNPE := false|ENABLE_SNPE := true|" api/src/main/jni/Android-nnstreamer-prebuilt.mk
    sed -i "s|ENABLE_SNPE := false|ENABLE_SNPE := true|" api/src/main/jni/Android.mk
    sed -i "$ a SNPE_EXT_LIBRARY_PATH=src/main/jni/snpe/lib/ext" gradle.properties

    mkdir -p api/src/main/jni/snpe/lib/ext/arm64-v8a
    cp -r $SNPE_DIRECTORY/include api/src/main/jni/snpe
    cp $SNPE_DIRECTORY/lib/aarch64-android-clang6.0/libSNPE.so api/src/main/jni/snpe/lib

    # Copy external so files for SNPE
    cp $SNPE_DIRECTORY/lib/aarch64-android-clang6.0/lib*dsp*.so api/src/main/jni/snpe/lib/ext/arm64-v8a
    cp $SNPE_DIRECTORY/lib/aarch64-android-clang6.0/libhta.so api/src/main/jni/snpe/lib/ext/arm64-v8a
    cp $SNPE_DIRECTORY/lib/dsp/libsnpe*.so api/src/main/jni/snpe/lib/ext/arm64-v8a
fi

# Update PyTorch option
if [[ $enable_pytorch == "yes" ]]; then
    sed -i "s|ENABLE_PYTORCH := false|ENABLE_PYTORCH := true|" api/src/main/jni/Android.mk
    sed -i "s|PYTORCH_VERSION := 1.8.0|PYTORCH_VERSION := $pytorch_ver|" api/src/main/jni/Android-pytorch.mk
    tar -xJf ./external/pytorch-$pytorch_ver.tar.xz -C ./api/src/main/jni
fi

# Update tf-lite option
if [[ $enable_tflite == "yes" ]]; then
    sed -i "s|ENABLE_TF_LITE := false|ENABLE_TF_LITE := true|" api/src/main/jni/Android-nnstreamer-prebuilt.mk
    sed -i "s|ENABLE_TF_LITE := false|ENABLE_TF_LITE := true|" api/src/main/jni/Android.mk
    sed -i "s|TFLITE_VERSION := 1.13.1|TFLITE_VERSION := $tf_lite_ver|" api/src/main/jni/Android-tensorflow-lite.mk
    tar -xJf ./external/tensorflow-lite-$tf_lite_ver.tar.xz -C ./api/src/main/jni
fi


if [[ $enable_decoder_flatbuf == "yes" ]]; then
    sed -i "s|ENABLE_DECODER_FLATBUF := false|ENABLE_DECODER_FLATBUF := true|" api/src/main/jni/Android.mk
    sed -i "s|FLATBUF_VER := @FLATBUF_VER@|FLATBUF_VER := ${decoder_flatbuf_ver}|" api/src/main/jni/Android-dec-flatbuf.mk
    wget --directory-prefix=./external https://raw.githubusercontent.com/nnstreamer/nnstreamer-android-resource/master/external/flatbuffers-${decoder_flatbuf_ver}.tar.xz
    tar -xJf ./external/flatbuffers-${decoder_flatbuf_ver}.tar.xz -C ./api/src/main/jni
fi

# Add dependency for release
if [[ $release_bintray == "yes" ]]; then
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
    website = 'https://github.com/nnstreamer/nnstreamer'\n\
    issueTracker = 'https://github.com/nnstreamer/nnstreamer/issues'\n\
    repository = 'https://github.com/nnstreamer/nnstreamer.git'\n\
}|" api/build.gradle
fi

# If build option is single-shot only, remove unnecessary files.
if [[ $nnstreamer_api_option == "single" ]]; then
    rm ./api/src/main/java/org/nnsuite/nnstreamer/CustomFilter.java
    rm ./api/src/main/java/org/nnsuite/nnstreamer/Pipeline.java
    rm ./api/src/androidTest/java/org/nnsuite/nnstreamer/APITestCustomFilter.java
    rm ./api/src/androidTest/java/org/nnsuite/nnstreamer/APITestPipeline.java
fi

echo "Starting gradle build for Android library."

# Build Android library.
chmod +x gradlew
sh ./gradlew api:build

# Check if build procedure is done.
nnstreamer_android_api_lib=./api/build/outputs/aar/api-release.aar

android_lib_build_res=1
if [[ -e "$nnstreamer_android_api_lib" ]]; then
    today=$(date "+%Y-%m-%d")
    android_lib_build_res=0

    # Prepare native libraries and header files for C-API
    unzip $nnstreamer_android_api_lib -d aar_extracted

    mkdir -p main/jni/nnstreamer/lib
    mkdir -p main/jni/nnstreamer/include

    # assets
    if [[ $include_assets == "yes" ]]; then
        mkdir -p main/assets
        cp -r aar_extracted/assets/* main/assets
    fi

    # native libraries and mk files
    cp -r aar_extracted/jni/* main/jni/nnstreamer/lib
    cp api/src/main/jni/*-prebuilt.mk main/jni

    # header for C-API
    cp $nnstreamer_dir/api/capi/include/nnstreamer.h main/jni/nnstreamer/include
    cp $nnstreamer_dir/api/capi/include/nnstreamer-single.h main/jni/nnstreamer/include
    cp $nnstreamer_dir/api/capi/include/platform/ml-api-common.h main/jni/nnstreamer/include

    # header for plugin
    if [[ $nnstreamer_api_option != "single" ]]; then
        cp $nnstreamer_dir/gst/nnstreamer/include/*.h main/jni/nnstreamer/include
        cp $nnstreamer_dir/gst/nnstreamer/include/*.hh main/jni/nnstreamer/include
        cp $nnstreamer_dir/ext/nnstreamer/tensor_filter/tensor_filter_cpp.hh main/jni/nnstreamer/include
    fi

    nnstreamer_native_files="$nnstreamer_lib_name-native-$today.zip"
    zip -r $nnstreamer_native_files main

    # Prepare static libs in ndk build
    mkdir -p ndk_static/debug
    mkdir -p ndk_static/release

    cp api/build/intermediates/ndkBuild/debug/obj/local/$target_abi/*.a ndk_static/debug
    cp api/build/intermediates/ndkBuild/release/obj/local/$target_abi/*.a ndk_static/release

    nnstreamer_static_libs="$nnstreamer_lib_name-static-libs-$today.zip"
    zip -r $nnstreamer_static_libs ndk_static

    rm -rf aar_extracted main ndk_static

    # Upload to jcenter
    if [[ $release_bintray == "yes" ]]; then
        echo "Upload NNStreamer library to Bintray."
        sh ./gradlew api:bintrayUpload -PbintrayUser=$bintray_user_name -PbintrayKey=$bintray_user_key -PdryRun=false
    fi

    # Run instrumentation test
    if [[ $run_test == "yes" ]]; then
        echo "Run instrumentation test."
        sh ./gradlew api:connectedCheck

        test_result="$nnstreamer_lib_name-test-$today.zip"
        zip -r $test_result api/build/reports
        cp $test_result $result_dir

        test_summary=$(sed -n "/<div class=\"percent\">/p" api/build/reports/androidTests/connected/index.html)
        if [[ $test_summary != "<div class=\"percent\">100%</div>" ]]; then
            echo "Instrumentation test failed."
            android_lib_build_res=1
        fi
    fi

    # Copy build result
    if [[ $android_lib_build_res == 0 ]]; then
        echo "Build procedure is done, copy NNStreamer library to $result_dir directory."

        cp $nnstreamer_android_api_lib $result_dir/$nnstreamer_lib_name-$today.aar
        cp $nnstreamer_native_files $result_dir
        cp $nnstreamer_static_libs $result_dir
    fi
else
    echo "Failed to build NNStreamer library."
fi

popd

# Remove build directory
rm -rf $build_dir

popd
cd ${nnstreamer_dir} && find -name nnstreamer_version.h -delete

# exit with success/failure status
exit $android_lib_build_res
