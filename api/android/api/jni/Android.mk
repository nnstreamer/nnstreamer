LOCAL_PATH := $(call my-dir)

ifndef GSTREAMER_ROOT_ANDROID
$(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
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

#------------------------------------------------------
# external libs
#------------------------------------------------------
#include $(LOCAL_PATH)/Android-tensorflow-lite.mk
include $(LOCAL_PATH)/Android-tensorflow-lite-prebuilt.mk

#------------------------------------------------------
# nnstreamer
#------------------------------------------------------
include $(LOCAL_PATH)/Android-nnstreamer.mk

#------------------------------------------------------
# native code for api
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := nnstreamer-native
LOCAL_SRC_FILES := nnstreamer-native-api.c \
    nnstreamer-native-customfilter.c \
    nnstreamer-native-pipeline.c \
    nnstreamer-native-singleshot.c
LOCAL_CFLAGS += -O2 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_C_INCLUDES := $(NNSTREAMER_INCLUDES) $(NNSTREAMER_CAPI_INCLUDES)
LOCAL_STATIC_LIBRARIES := nnstreamer tensorflow-lite cpufeatures
LOCAL_SHARED_LIBRARIES := gstreamer_android
LOCAL_LDLIBS := -llog -landroid

include $(BUILD_SHARED_LIBRARY)

GSTREAMER_NDK_BUILD_PATH  := $(GSTREAMER_ROOT)/share/gst-android/ndk-build/
include $(GSTREAMER_NDK_BUILD_PATH)/plugins.mk
# add necessary gstreamer plugins
GSTREAMER_PLUGINS         := $(GSTREAMER_PLUGINS_CORE) \
    $(GSTREAMER_PLUGINS_CODECS) \
    $(GSTREAMER_PLUGINS_ENCODING) \
    $(GSTREAMER_PLUGINS_PLAYBACK) \
    $(GSTREAMER_PLUGINS_VIS) \
    $(GSTREAMER_PLUGINS_SYS) \
    $(GSTREAMER_PLUGINS_EFFECTS) \
    $(GSTREAMER_PLUGINS_CAPTURE) \
    $(GSTREAMER_PLUGINS_CODECS_GPL) \
    $(GSTREAMER_PLUGINS_CODECS_RESTRICTED) \
    $(GSTREAMER_PLUGINS_NET_RESTRICTED) \
    $(GSTREAMER_PLUGINS_GES)
GSTREAMER_EXTRA_DEPS      := gstreamer-video-1.0 gstreamer-audio-1.0 gstreamer-app-1.0 gobject-2.0
GSTREAMER_EXTRA_LIBS      := -liconv -lcairo
include $(GSTREAMER_NDK_BUILD_PATH)/gstreamer-1.0.mk

$(call import-module, android/cpufeatures)
