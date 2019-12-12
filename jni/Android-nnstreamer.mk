LOCAL_PATH := $(call my-dir)
#
# Geunsik Lim <geunsik.lim@samsung.com>
# This configuration file is to generate nnstreamer library.
#
# Step1: Required packages
# - Software Platform: Android 7.0 (Nougat) + ARM 64bit
#   - NDK version: r12b
#   - API Level: 24
# - Gstreamer 1.12
#
# Step2: Append a path of Android Gstreamer and Android NDK as following:
# vi ~/.bashrc
# # gst-root-android, Android NDK/SDK
# export GSTREAMER_ROOT_ANDROID=/work/taos/gst-android/gst_root_android
# export ANDROID_NDK=/work/taos/gst-android/android-ndk-r12b
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

# Common definition for NNStreamer
include $(LOCAL_PATH)/nnstreamer.mk

# Define shared libraries that are required by a gstreamer plug-in.
define shared_lib_common
    include $(CLEAR_VARS)
    LOCAL_MODULE := $(1)
    LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/lib$(1).so
    include $(PREBUILT_SHARED_LIBRARY)
endef

# Define shared libraries that are used as a gstreamer plug-in.
define shared_lib_gst
    include $(CLEAR_VARS)
    LOCAL_MODULE := $(1)
    LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/lib$(1).so
    include $(PREBUILT_SHARED_LIBRARY)
endef

# Describe shared libraries that are needed to run this application.

so_names_common := gstreamer-1.0 gstbase-1.0 gstvideo-1.0 glib-2.0 \
                   gobject-2.0 intl z bz2 orc-0.4 gmodule-2.0 ffi gsttag-1.0 iconv \
                   gstapp-1.0 png16 gstbadbase-1.0 gio-2.0 pangocairo-1.0 \
                   pangoft2-1.0 pango-1.0 gthread-2.0 cairo pixman-1 fontconfig expat freetype \
                   gstbadvideo-1.0 gstcontroller-1.0 jpeg graphene-1.0 gstpbutils-1.0 gstgl-1.0 \
                   gstallocators-1.0 gstbadallocators-1.0 harfbuzz

ifeq ($(NO_AUDIO), false)
so_names_common += gstaudio-1.0 gstbadaudio-1.0
endif

$(foreach item,$(so_names_common),$(eval $(call shared_lib_common,$(item))))

so_names_gst := gstcoreelements gstcoretracers gstadder gstapp \
                gstpango gstrawparse gsttypefindfunctions gstvideoconvert gstvideorate \
                gstvideoscale gstvideotestsrc gstvolume gstautodetect gstvideofilter gstopengl \
                gstopensles gstcompositor gstpng gstmultifile

ifeq ($(NO_AUDIO), false)
so_names_gst += gstaudioconvert gstaudiomixer gstaudiorate gstaudioresample gstaudiotestsrc
endif

$(foreach item,$(so_names_gst),$(eval $(call shared_lib_gst,$(item))))

include $(CLEAR_VARS)

# Please keep the pthread and openmp library for checking a compatibility
LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -std=c++11 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

ifeq ($(NO_AUDIO), true)
LOCAL_CFLAGS += -DNO_AUDIO
LOCAL_CXXFLAGS += -DNO_AUDIO
endif

LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_MODULE        := nnstreamer
LOCAL_SRC_FILES     := $(NNSTREAMER_COMMON_SRCS) $(NNSTREAMER_PLUGINS_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

BUILDING_BLOCK_LIST := gstreamer-1.0 glib-2.0 gobject-2.0 intl gstcoreelements \
gstapp pixman-1 fontconfig expat freetype \
gstvideoconvert gstvideorate gstvideoscale \
gmodule-2.0 iconv png16 gstpng gstmultifile gio-2.0 \
gstbase-1.0 gstvideo-1.0 tag-1.0 orc app-1.0 badbase-1.0 gthread \
cairo pixman gstbadvideo gstcontroller jpeg gstpbutils gstallocators \
bz2 harfbuzz

ifeq ($(NO_AUDIO), false)
BUILDING_BLOCK_LIST += gstaudio-1.0 gstbadaudio-1.0 gstaudioconvert gstaudiomixer gstaudiorate gstaudioresample gstaudiotestsrc
endif

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

LOCAL_SHARED_LIBRARIES := $(BUILDING_BLOCK_LIST)

include $(BUILD_SHARED_LIBRARY)
