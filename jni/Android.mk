LOCAL_PATH := $(call my-dir)
#
# Geunsik Lim <geunsik.lim@samsung.com>
#
# Step1: Required packages
# - Software Platform: Android 7.0 (Nougat) + ARM 64bit
#   - NDK version: r12b
#   - API Level: 24
# - Gstreamer 1.12
#
# Step2: Append a path of Android Gstreamer and Android NDK as following:
# ubuntu16.04$> vi ~/.bashrc
# # gst-root-android, Android NDK/SDK
# export GSTREAMER_ROOT_ANDROID=/work/taos/gst-android/gst_root_android
# export ANDROID_NDK=/work/taos/gst-android/android-ndk-r12b
# export PATH=$ANDROID_NDK:$PATH
#
# Step3: Build NNStreamer for Android platform
# ubuntu16.04$> ndk-build -j$(nproc)
#
# Step4: Install nnstreamer into Android target device
# for i in ../libs/arm64-v8a/*.so ; do echo "$i" ; adb push "$i" /data/nnstreamer/libgstreamer/;  done;
# adb shell mv /data/nnstreamer/libgstreamer/libnnstreamer.so    /data/nnstreamer/libnnstreamer/
# adb shell ln -s /data/nnstreamer/libgstreamer/libharfbuzz.so   /data/nnstreamer/libgstreamer/libharfbuzz.so.0
# adb shell ln -s /data/nnstreamer/libgstreamer/libbz2.so        /data/nnstreamer/libgstreamer/libbz2.so.1.0
#
# readelf -d ../libs/arm64-v8a/test
# adb push ../libs/arm64-v8a/test  /data/nnstreamer/
#

NNSTREAMER_VERSION := 0.1.1
TARGET_ARCH_ABI    := arm64-v8a

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

# Describe shared libraries that are needed to run this application.
include $(CLEAR_VARS)
LOCAL_MODULE := gstreamer-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstreamer-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstbase-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstbase-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstaudio-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstaudio-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvideo-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstvideo-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := glib-2.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libglib-2.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gobject-2.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgobject-2.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := intl
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libintl.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := z
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libz.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := bz2
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libbz2.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstcoreelements
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstcoreelements.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstcoretracers
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstcoretracers.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstadder
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstadder.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := orc
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/liborc-0.4.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstapp
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstapp.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstaudioconvert
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstaudioconvert.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstaudiomixer
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstaudiomixer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstaudiorate
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstaudiorate.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstaudioresample
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstaudioresample.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstaudiotestsrc
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstaudiotestsrc.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstgio
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstgio.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstpango
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstpango.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstrawparse
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstrawparse.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gsttypefindfunctions
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgsttypefindfunctions.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvideoconvert
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstvideoconvert.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvideorate
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstvideorate.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvideoscale
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstvideoscale.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvideotestsrc
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstvideotestsrc.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvolume
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstvolume.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstautodetect
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstautodetect.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstvideofilter
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstvideofilter.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstopengl
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstopengl.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstopensles
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstopensles.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gmodule-2.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgmodule-2.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstcompositor
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstcompositor.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ffi
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libffi.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := tag-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgsttag-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := iconv
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libiconv.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := app-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstapp-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := png
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libpng16.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstpng
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstpng.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := badaudio
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstbadaudio-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := badbase-1.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstbadbase-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gio-2.0
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgio-2.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pangocairo
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libpangocairo-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pangoft2
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libpangoft2-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pango
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libpango-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gthread
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgthread-2.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := cairo
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libcairo.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pixman
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libpixman-1.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := fontconfig
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libfontconfig.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := expat
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libexpat.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := freetype
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libfreetype.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstbadvideo
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstbadvideo-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstcontroller
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstcontroller-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := jpeg
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libjpeg.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := graphene
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgraphene-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstpbutils
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstpbutils-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstgl
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstgl-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstbadallocators
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstbadallocators-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := gstallocators
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libgstallocators-1.0.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := harfbuzz
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/libharfbuzz.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := multifile
LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/libgstmultifile.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

# Please keep the pthread and openmp library for checking a compatibility
LOCAL_ARM_NEON := true
LOCAL_CFLAGS   += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS += -std=c++11
LOCAL_CFLAGS   += -pthread -fopenmp

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

BUILDING_BLOCK_LIST := gstreamer-1.0 glib-2.0 gobject-2.0 intl gstcoreelements gstcoretracers gstadder \
gstapp gstaudioconvert gstaudiomixer gstaudioresample gstaudiorate gstaudioresample gstaudiotestsrc gstgio \
gstpango gstrawparse gsttypefindfunctions gstvideoconvert gstvideorate gstvideoscale gstvideotestsrc \
gstvolume gstautodetect gstvideofilter gstopengl gstopensles gmodule-2.0 gstcompositor ffi iconv png multifile \
gstbase-1.0 gstaudio-1.0 gstvideo-1.0 tag-1.0 orc app-1.0 badaudio badbase-1.0 gio-2.0 pangocairo  pango gthread \
 cairo pixman fontconfig expat gstbadvideo gstcontroller jpeg graphene gstpbutils gstgl gstbadallocators \
gstallocators harfbuzz bz2

LOCAL_C_INCLUDES += $(GSTREAMER_ROOT)/include/gstreamer-1.0 \
     $(GSTREAMER_ROOT)/include/glib-2.0 \
     $(GSTREAMER_ROOT)/lib/glib-2.0/include \
     $(GSTREAMER_ROOT)/include

LOCAL_SHARED_LIBRARIES := $(BUILDING_BLOCK_LIST)

include $(BUILD_SHARED_LIBRARY)

# In case of Android ARM 64bit environment, the default path of linker is "/data/nnstreamer/".
# We use the "tests/nnstreamer_repo_dynamicity/tensor_repo_dynamic_test.c" file as a test application.
# This application is dependent on 'multifilesrc' and 'png' element that are provided by Gstreamer.
include $(CLEAR_VARS)
LOCAL_MODULE    := test
LOCAL_CFLAGS    += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_SRC_FILES += ../tests/nnstreamer_repo_dynamicity/tensor_repo_dynamic_test.c 
LOCAL_CXXFLAGS  += -std=c++11
LOCAL_LDFLAGS   := -fPIE -pie -Wl,-dynamic-linker,/data/nnstreamer/libandroid/linker64

NNSTREAMER_GST_HOME    := ../gst/nnstreamer

LOCAL_C_INCLUDES       := $(NNSTREAMER_GST_HOME)
LOCAL_SHARED_LIBRARIES := $(BUILDING_BLOCK_LIST) nnstreamer

LOCAL_C_INCLUDES += $(GSTREAMER_ROOT)/include/gstreamer-1.0 \
     $(GSTREAMER_ROOT)/include/glib-2.0 \
     $(GSTREAMER_ROOT)/lib/glib-2.0/include \
     $(GSTREAMER_ROOT)/include

GSTREAMER_ANDROID_INCLUDE := $(GSTREAMER_ROOT)/include

include $(BUILD_EXECUTABLE)
