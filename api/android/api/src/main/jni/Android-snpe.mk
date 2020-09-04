#------------------------------------------------------
# SNPE (The Snapdragon Neural Processing Engine)
#
# This mk file defines snpe module with prebuilt shared library.
# (snpe-sdk, arm64-v8a only)
# See Qualcomm Neural Processing SDK for AI (https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) for the details.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

SNPE_DIR := $(LOCAL_PATH)/snpe
SNPE_INCLUDES := $(SNPE_DIR)/include/zdl/

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
SNPE_LIB_PATH := $(SNPE_DIR)/lib
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

#------------------------------------------------------
# snpe-sdk (prebuilt shared library)
#------------------------------------------------------
include $(LOCAL_PATH)/Android-snpe-prebuilt.mk

#------------------------------------------------------
# tensor-filter sub-plugin for snpe
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := snpe-subplugin
LOCAL_SRC_FILES := $(NNSTREAMER_FILTER_SNPE_SRCS)
LOCAL_CXXFLAGS := -std=c++11 -O3 -fPIC -frtti -fexceptions
LOCAL_C_INCLUDES := $(SNPE_INCLUDES) $(NNSTREAMER_INCLUDES) $(GST_HEADERS_COMMON)
LOCAL_STATIC_LIBRARIES := nnstreamer
LOCAL_SHARED_LIBRARIES := $(SNPE_PREBUILT_LIBS)

include $(BUILD_STATIC_LIBRARY)
