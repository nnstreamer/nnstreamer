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

NNSTREAMER_VERSION := 0.1.3
NO_AUDIO := false

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
LOCAL_CXXFLAGS      += -std=c++11
LOCAL_CFLAGS        += -pthread -fopenmp

ifeq ($(NO_AUDIO), true)
LOCAL_CFLAGS += -DNO_AUDIO
LOCAL_CXXFLAGS += -DNO_AUDIO
endif

LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_MODULE	    := nnstreamer
NNSTREAMER_GST_HOME := ../gst/nnstreamer
NNSTREAMER_GST_TEST := ../nnstreamer_example/
LOCAL_SRC_FILES := $(NNSTREAMER_GST_HOME)/nnstreamer.c \
	$(NNSTREAMER_GST_HOME)/nnstreamer_conf.c \
	$(NNSTREAMER_GST_HOME)/nnstreamer_subplugin.c \
	$(NNSTREAMER_GST_HOME)/tensor_common.c \
	$(NNSTREAMER_GST_HOME)/tensor_repo.c \
	$(NNSTREAMER_GST_HOME)/tensor_converter/tensor_converter.c \
	$(NNSTREAMER_GST_HOME)/tensor_aggregator/tensor_aggregator.c \
	$(NNSTREAMER_GST_HOME)/tensor_decoder/tensordec.c \
	$(NNSTREAMER_GST_HOME)/tensor_demux/gsttensordemux.c \
	$(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter.c \
	$(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_custom.c \
	$(NNSTREAMER_GST_HOME)/tensor_merge/gsttensormerge.c \
	$(NNSTREAMER_GST_HOME)/tensor_mux/gsttensormux.c \
	$(NNSTREAMER_GST_HOME)/tensor_reposink/tensor_reposink.c \
	$(NNSTREAMER_GST_HOME)/tensor_reposrc/tensor_reposrc.c \
	$(NNSTREAMER_GST_HOME)/tensor_saveload/tensor_load.c \
	$(NNSTREAMER_GST_HOME)/tensor_sink/tensor_sink.c \
	$(NNSTREAMER_GST_HOME)/tensor_split/gsttensorsplit.c \
	$(NNSTREAMER_GST_HOME)/tensor_transform/tensor_transform.c

LOCAL_C_INCLUDES := $(NNSTREAMER_GST_HOME)/ \
	$(NNSTREAMER_GST_HOME)/tensor_converter/ \
	$(NNSTREAMER_GST_HOME)/tensor_aggregator/ \
	$(NNSTREAMER_GST_HOME)/tensor_decoder/ \
	$(NNSTREAMER_GST_HOME)/tensor_demux/ \
	$(NNSTREAMER_GST_HOME)/tensor_filter/ \
	$(NNSTREAMER_GST_HOME)/tensor_merge/ \
	$(NNSTREAMER_GST_HOME)/tensor_mux/ \
	$(NNSTREAMER_GST_HOME)/tensor_reposink/ \
	$(NNSTREAMER_GST_HOME)/tensor_reposrc/ \
	$(NNSTREAMER_GST_HOME)/tensor_saveload/ \
	$(NNSTREAMER_GST_HOME)/tensor_sink/ \
	$(NNSTREAMER_GST_HOME)/tensor_split/ \
	$(NNSTREAMER_GST_HOME)/tensor_transform/

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

LOCAL_C_INCLUDES += $(GSTREAMER_ROOT)/include/gstreamer-1.0 \
     $(GSTREAMER_ROOT)/include/glib-2.0 \
     $(GSTREAMER_ROOT)/lib/glib-2.0/include \
     $(GSTREAMER_ROOT)/include

LOCAL_SHARED_LIBRARIES := $(BUILDING_BLOCK_LIST)

include $(BUILD_SHARED_LIBRARY)
