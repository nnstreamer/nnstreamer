#------------------------------------------------------
# SNAP (Samsung Neural Acceleration Platform)
#
# This mk file defines snap module with prebuilt shared library.
# (snap-sdk, arm64-v8a only)
# See Samsung Neural SDK (https://developer.samsung.com/neural) for the details.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

SNAP_DIR := $(LOCAL_PATH)/snap
SNAP_INCLUDES := $(SNAP_DIR)/include

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
SNAP_LIB_PATH := $(SNAP_DIR)/lib
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

#------------------------------------------------------
# snap-sdk (prebuilt shared library)
#------------------------------------------------------
include $(LOCAL_PATH)/Android-snap-prebuilt.mk

#------------------------------------------------------
# tensor-filter sub-plugin for snap
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := snap-subplugin
LOCAL_SRC_FILES := $(NNSTREAMER_FILTER_SNAP_SRCS)
LOCAL_CXXFLAGS := -std=c++11 -O3 -fPIC -frtti -fexceptions $(NNS_API_FLAGS)
LOCAL_C_INCLUDES := $(SNAP_INCLUDES) $(NNSTREAMER_INCLUDES) $(GST_HEADERS_COMMON)
LOCAL_STATIC_LIBRARIES := nnstreamer
LOCAL_SHARED_LIBRARIES := $(SNAP_PREBUILT_LIBS)

include $(BUILD_STATIC_LIBRARY)
