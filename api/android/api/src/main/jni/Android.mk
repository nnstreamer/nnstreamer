LOCAL_PATH := $(call my-dir)

ifndef GSTREAMER_ROOT_ANDROID
$(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
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
# API build option
#------------------------------------------------------
include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

NNSTREAMER_API_OPTION := all

# tensorflow-lite (nnstreamer tf-lite subplugin)
ENABLE_TF_LITE := false

# SNAP (Samsung Neural Acceleration Platform)
ENABLE_SNAP := false

# NNFW (On-device neural network inference framework, Samsung Research)
ENABLE_NNFW := false

# SNPE (Snapdragon Neural Processing Engine)
ENABLE_SNPE := false

ifeq ($(ENABLE_SNAP),true)
  ifeq ($(ENABLE_SNPE),true)
   $(error DO NOT enable SNAP and SNPE both. The app would fail to use DSP or NPU runtime.)
  endif
endif

# Common options
NNS_API_INCLUDES := \
    $(NNSTREAMER_INCLUDES) \
    $(NNSTREAMER_CAPI_INCLUDES) \
    $(GST_HEADERS_COMMON)

NNS_API_FLAGS := -DVERSION=\"$(NNSTREAMER_VERSION)\"
NNS_SUBPLUGINS :=

ifeq ($(NNSTREAMER_API_OPTION),single)
NNS_API_FLAGS += -DNNS_SINGLE_ONLY=1
endif

#------------------------------------------------------
# external libs and sub-plugins
#------------------------------------------------------
ifeq ($(ENABLE_TF_LITE),true)
NNS_API_FLAGS += -DENABLE_TENSORFLOW_LITE=1
NNS_SUBPLUGINS += tensorflow-lite-subplugin

include $(LOCAL_PATH)/Android-tensorflow-lite.mk
endif

ifeq ($(ENABLE_SNAP),true)
NNS_API_FLAGS += -DENABLE_SNAP=1
NNS_SUBPLUGINS += snap-subplugin

include $(LOCAL_PATH)/Android-snap.mk
endif

ifeq ($(ENABLE_NNFW),true)
NNS_API_FLAGS += -DENABLE_NNFW=1
NNS_SUBPLUGINS += nnfw-subplugin

include $(LOCAL_PATH)/Android-nnfw.mk
endif

ifeq ($(ENABLE_SNPE),true)
NNS_API_FLAGS += -DENABLE_SNPE=1
NNS_SUBPLUGINS += snpe-subplugin

include $(LOCAL_PATH)/Android-snpe.mk
endif

#------------------------------------------------------
# nnstreamer
#------------------------------------------------------
include $(LOCAL_PATH)/Android-nnstreamer.mk

# Remove any duplicates.
NNS_SUBPLUGINS := $(sort $(NNS_SUBPLUGINS))

#------------------------------------------------------
# native code for api
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := nnstreamer-native

LOCAL_SRC_FILES := \
    nnstreamer-native-api.c \
    nnstreamer-native-singleshot.c

ifneq ($(NNSTREAMER_API_OPTION),single)
LOCAL_SRC_FILES += \
    nnstreamer-native-customfilter.c \
    nnstreamer-native-pipeline.c
endif

LOCAL_C_INCLUDES := $(NNS_API_INCLUDES)
LOCAL_CFLAGS := -O3 -fPIC $(NNS_API_FLAGS)
LOCAL_STATIC_LIBRARIES := nnstreamer $(NNS_SUBPLUGINS)
LOCAL_SHARED_LIBRARIES := gstreamer_android
LOCAL_LDLIBS := -llog -landroid

ifneq ($(NNSTREAMER_API_OPTION),single)
# For amcsrc element
LOCAL_LDLIBS += -lmediandk
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

ifeq ($(NNSTREAMER_API_OPTION),all)
GSTREAMER_EXTRA_LIBS += -lcairo
endif

GSTREAMER_INCLUDE_FONTS := no
GSTREAMER_INCLUDE_CA_CERTIFICATES := no

include $(GSTREAMER_NDK_BUILD_PATH)/gstreamer-1.0.mk

#------------------------------------------------------
# NDK cpu-features
#------------------------------------------------------
$(call import-module, android/cpufeatures)
