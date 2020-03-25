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
# API build option
#------------------------------------------------------
NNSTREAMER_API_OPTION := all

# tensorflow-lite (nnstreamer tf-lite subplugin)
ENABLE_TF_LITE := false

# SNAP (Samsung Neural Acceleration Platform)
ENABLE_SNAP := false

NNS_API_FLAGS := -DVERSION=\"$(NNSTREAMER_VERSION)\"
NNS_API_STATIC_LIBS :=

ifeq ($(NNSTREAMER_API_OPTION),single)
NNS_API_FLAGS += -DNNS_SINGLE_ONLY=1
endif

#------------------------------------------------------
# external libs
#------------------------------------------------------
ifeq ($(ENABLE_TF_LITE),true)
NNS_API_FLAGS += -DENABLE_TENSORFLOW_LITE=1
# define types in tensorflow-lite sub-plugin. This assumes tensorflow-lite >= 1.13 (older versions don't have INT8/INT16)
NNS_API_FLAGS += -DTFLITE_INT8=1 -DTFLITE_INT16=1
NNS_API_STATIC_LIBS += tensorflow-lite cpufeatures

include $(LOCAL_PATH)/Android-tensorflow-lite.mk
endif

ifeq ($(ENABLE_SNAP),true)
NNS_API_FLAGS += -DENABLE_SNAP=1
NNS_API_STATIC_LIBS += snap

include $(LOCAL_PATH)/Android-snap.mk
endif

#------------------------------------------------------
# nnstreamer
#------------------------------------------------------
include $(LOCAL_PATH)/Android-nnstreamer.mk

# Remove any duplicates.
NNS_API_STATIC_LIBS := $(sort $(NNS_API_STATIC_LIBS))

#------------------------------------------------------
# native code for api
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := nnstreamer-native
LOCAL_SRC_FILES := nnstreamer-native-api.c \
    nnstreamer-native-singleshot.c
LOCAL_CFLAGS += -O2 $(NNS_API_FLAGS)
LOCAL_C_INCLUDES := $(NNSTREAMER_INCLUDES) $(NNSTREAMER_CAPI_INCLUDES)
LOCAL_STATIC_LIBRARIES := nnstreamer $(NNS_API_STATIC_LIBS)
LOCAL_SHARED_LIBRARIES := gstreamer_android
LOCAL_LDLIBS := -llog -landroid

ifneq ($(NNSTREAMER_API_OPTION),single)
LOCAL_SRC_FILES += \
    nnstreamer-native-customfilter.c \
    nnstreamer-native-pipeline.c
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

GSTREAMER_INCLUDE_FONTS := no
GSTREAMER_INCLUDE_CA_CERTIFICATES := no

include $(GSTREAMER_NDK_BUILD_PATH)/gstreamer-1.0.mk

#------------------------------------------------------
# NDK cpu-features
#------------------------------------------------------
ifeq ($(filter cpufeatures,$(NNS_API_STATIC_LIBS)),cpufeatures)
$(call import-module, android/cpufeatures)
endif
