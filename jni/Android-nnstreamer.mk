LOCAL_PATH := $(call my-dir)
#
# Geunsik Lim <geunsik.lim@samsung.com>
# This configuration file is to generate nnstreamer library.
#
# Step1: Required packages
# - Software Platform: Android 7.0 (Nougat) + ARM 64bit
#   - NDK version: r25c
#   - API Level: 33
# - Gstreamer 1.24
#
# Step2: Append a path of Android Gstreamer and Android NDK as following:
# vi ~/.bashrc
# # gst-root-android, Android NDK/SDK
# export GSTREAMER_ROOT_ANDROID=/work/taos/gst-android/gst_root_android
# export ANDROID_NDK=/work/taos/gst-android/android-ndk-r25c
# export PATH=$ANDROID_NDK:$PATH
#
# Step3: Build NNStreamer for Android platform
# ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android-nnstreamer.mk NDK_APPLICATION_MK=./Application.mk -j$(nproc)
#
# Step4: Install the nnstreamer library into Android target device
# for i in ./libs/arm64-v8a/*.so ; do echo "$i" ; adb push "$i" /data/nnstreamer/libgstreamer/; done;
# adb shell ln -s /data/nnstreamer/libgstreamer/libharfbuzz.so   /data/nnstreamer/libgstreamer/libharfbuzz.so.0
# adb shell ln -s /data/nnstreamer/libgstreamer/libbz2.so        /data/nnstreamer/libgstreamer/libbz2.so.1.0
# adb shell mv /data/nnstreamer/libgstreamer/libnnstreamer.so    /data/nnstreamer/libnnstreamer/
# cp ./libs/arm64-v8a/libnnstreamer.so $GSTREAMER_ROOT_ANDROID/arm64/lib/gstreamer-1.0/
#

ifndef NNSTREAMER_ROOT
NNSTREAMER_ROOT := $(LOCAL_PATH)/..
endif

# Do not specify "TARGET_ARCH_ABI" in this file. If you want to append additional architecture,
# Please append an architecture name behind "APP_ABI" in Application.mk file.

ifeq ($(TARGET_ARCH_ABI),armeabi)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/arm
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/armv7
else ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/arm64
else ifeq ($(TARGET_ARCH_ABI),x86)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/x86
else ifeq ($(TARGET_ARCH_ABI),x86_64)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/x86_64
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

GSTREAMER_NDK_BUILD_PATH := $(GSTREAMER_ROOT)/share/gst-android/ndk-build
include $(LOCAL_PATH)/Android-gst-plugins.mk

GST_BLOCKED_PLUGINS      := \
        fallbackswitch livesync rsinter rstracers \
        threadshare togglerecord cdg claxon dav1d rsclosedcaption \
        ffv1 fmp4 mp4 gif hsv lewton rav1e json rspng regex textwrap textahead \
        aws hlssink3 ndi rsonvif raptorq reqwest rsrtp rsrtsp webrtchttp rswebrtc uriplaylistbin \
        rsaudiofx rsvideofx

GSTREAMER_PLUGINS        := $(filter-out $(GST_BLOCKED_PLUGINS), $(GST_REQUIRED_PLUGINS))
GSTREAMER_EXTRA_DEPS     := $(GST_REQUIRED_DEPS) glib-2.0 gio-2.0 gmodule-2.0
GSTREAMER_EXTRA_LIBS     := $(GST_REQUIRED_LIBS) -liconv

ifeq ($(NNSTREAMER_API_OPTION),all)
GSTREAMER_EXTRA_LIBS += -lcairo
endif

GSTREAMER_INCLUDE_FONTS := no
GSTREAMER_INCLUDE_CA_CERTIFICATES := no

include $(GSTREAMER_NDK_BUILD_PATH)/gstreamer-1.0.mk

# Common definition for NNStreamer
include $(LOCAL_PATH)/nnstreamer.mk

include $(CLEAR_VARS)

# Please keep the pthread and openmp library for checking a compatibility
LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -fPIC -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -fPIC -frtti -fexceptions -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

ifeq ($(NO_AUDIO), true)
LOCAL_CFLAGS += -DNO_AUDIO
LOCAL_CXXFLAGS += -DNO_AUDIO
endif

LOCAL_LDLIBS        := -llog -landroid
LOCAL_MODULE_TAGS   := optional

LOCAL_MODULE        := nnstreamer
LOCAL_SRC_FILES     := $(NNSTREAMER_COMMON_SRCS) $(NNSTREAMER_PLUGINS_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

LOCAL_SHARED_LIBRARIES := gstreamer_android

include $(BUILD_SHARED_LIBRARY)
