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

# SNAP (Samsung Neural Acceleration Platform)
ENABLE_SNAP := false

#------------------------------------------------------
# API build option
#------------------------------------------------------
NNSTREAMER_API_OPTION := all

#------------------------------------------------------
# external libs
#------------------------------------------------------
include $(LOCAL_PATH)/Android-tensorflow-lite-prebuilt.mk

ifeq ($(ENABLE_SNAP), true)
include $(LOCAL_PATH)/Android-snap.mk
endif

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
    nnstreamer-native-singleshot.c
LOCAL_CFLAGS += -O2 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_C_INCLUDES := $(NNSTREAMER_INCLUDES) $(NNSTREAMER_CAPI_INCLUDES)
LOCAL_STATIC_LIBRARIES := nnstreamer tensorflow-lite cpufeatures
LOCAL_SHARED_LIBRARIES := gstreamer_android
LOCAL_LDLIBS := -llog -landroid -lmediandk

ifeq ($(NNSTREAMER_API_OPTION),single)
LOCAL_CFLAGS += -DNNS_SINGLE_ONLY=1
else
LOCAL_SRC_FILES += \
    nnstreamer-native-customfilter.c \
    nnstreamer-native-pipeline.c
endif

ifeq ($(ENABLE_SNAP), true)
LOCAL_CFLAGS += -DENABLE_SNAP=1
LOCAL_STATIC_LIBRARIES += snap
endif

include $(BUILD_SHARED_LIBRARY)

#------------------------------------------------------
# gstreamer for android
#------------------------------------------------------
GSTREAMER_NDK_BUILD_PATH := $(GSTREAMER_ROOT)/share/gst-android/ndk-build/
include $(LOCAL_PATH)/Android-gst-plugins.mk

GSTREAMER_PLUGINS        := $(GST_REQUIRED_PLUGINS)
GSTREAMER_EXTRA_DEPS     := $(GST_REQUIRED_DEPS) gio-2.0 gmodule-2.0
GSTREAMER_EXTRA_LIBS     := $(GST_REQUIRED_LIBS) -liconv

GSTREAMER_INCLUDE_FONTS := no
GSTREAMER_INCLUDE_CA_CERTIFICATES := no

include $(GSTREAMER_NDK_BUILD_PATH)/gstreamer-1.0.mk

$(call import-module, android/cpufeatures)
