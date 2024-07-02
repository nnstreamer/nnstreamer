LOCAL_PATH := $(call my-dir)
#
# Geunsik Lim <geunsik.lim@samsung.com>
# This configuration file is to compile a test application
# using Gstreamer + NNstreamer library.
#
# Step1: Build a test application based on nnstreamer for Android platform
# ndk-build NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=./Android-app.mk NDK_APPLICATION_MK=./Application.mk -j$(nproc)
#
# Step2: Install a test application into Android target device
# readelf -d ./libs/arm64-v8a/{your-test-app}
# adb push   ./libs/arm64-v8a/{your-test-app} /data/nnstreamer/
#
# Step3: Run the test application
# adb shell
# target#> cd /data/nnstreamer/
# target#> ./{your-test-app}

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

# In case of Android ARM 64bit environment, the default path of linker is "/data/nnstreamer/".
# We use the "tests/nnstreamer_repo_dynamicity/tensor_repo_dynamic_test.c" file as a test application.
# This application is dependent on 'multifilesrc' and 'png' element that are provided by Gstreamer.
include $(CLEAR_VARS)
LOCAL_MODULE    := tensor_repo_dynamic_test
LOCAL_SRC_FILES += ../tests/nnstreamer_repo_dynamicity/tensor_repo_dynamic_test.c
LOCAL_CFLAGS    += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS  += -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_LDLIBS    += -llog

LOCAL_C_INCLUDES       := $(NNSTREAMER_INCLUDES)
LOCAL_SHARED_LIBRARIES := gstreamer_android

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

GSTREAMER_ANDROID_INCLUDE := $(GSTREAMER_ROOT)/include

include $(BUILD_EXECUTABLE)
