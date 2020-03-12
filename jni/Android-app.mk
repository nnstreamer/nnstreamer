LOCAL_PATH := $(call my-dir)
#
# Geunsik Lim <geunsik.lim@samsung.com>
# This configuration file is to compile a test application
# using Gstreamer + NNstreamer library.
#
# Step1: Build a test appliation based on nnstreamer for Android platform
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

CUSTOM_LINKER64    := -fPIE -pie -Wl,-dynamic-linker,/data/nnstreamer/libandroid/linker64

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
                gstopensles gstcompositor gstpng gstmultifile nnstreamer

ifeq ($(NO_AUDIO), false)
so_names_gst += gstaudioconvert gstaudiomixer gstaudiorate gstaudioresample gstaudiotestsrc
endif

$(foreach item,$(so_names_gst),$(eval $(call shared_lib_gst,$(item))))

BUILDING_BLOCK_LIST := gstreamer-1.0 glib-2.0 gobject-2.0 intl gstcoreelements \
gstapp pixman-1 fontconfig expat freetype \
gstvideoconvert gstvideorate gstvideoscale \
gmodule-2.0 iconv png16 gstpng gstmultifile gio-2.0 \
gstbase-1.0 gstvideo-1.0 tag-1.0 orc app-1.0 badbase-1.0 gthread \
cairo pixman gstbadvideo gstcontroller jpeg gstpbutils gstallocators \
bz2 harfbuzz nnstreamer


ifeq ($(NO_AUDIO), false)
BUILDING_BLOCK_LIST += gstaudio-1.0 gstbadaudio-1.0 gstaudioconvert gstaudiomixer gstaudiorate gstaudioresample gstaudiotestsrc
endif

# In case of Android ARM 64bit environment, the default path of linker is "/data/nnstreamer/".
# We use the "tests/nnstreamer_repo_dynamicity/tensor_repo_dynamic_test.c" file as a test application.
# This application is dependent on 'multifilesrc' and 'png' element that are provided by Gstreamer.
include $(CLEAR_VARS)
LOCAL_MODULE    := tensor_repo_dynamic_test
LOCAL_SRC_FILES += ../tests/nnstreamer_repo_dynamicity/tensor_repo_dynamic_test.c
LOCAL_CFLAGS    += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS  += -std=c++11 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_LDLIBS    := -llog
LOCAL_LDFLAGS   := $(CUSTOM_LINKER64)

LOCAL_C_INCLUDES       := $(NNSTREAMER_INCLUDES)
LOCAL_SHARED_LIBRARIES := $(BUILDING_BLOCK_LIST)

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

GSTREAMER_ANDROID_INCLUDE := $(GSTREAMER_ROOT)/include

include $(BUILD_EXECUTABLE)
