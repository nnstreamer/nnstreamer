#------------------------------------------------------
# NNFW (On-device neural network inference framework, which is developed by Samsung Research.)
# https://github.com/Samsung/ONE
#
# This mk file defines nnfw module with prebuilt shared library.
# (nnfw core libraries, arm64-v8a only)
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

NNFW_DIR := $(LOCAL_PATH)/nnfw
NNFW_INCLUDES := $(NNFW_DIR)/include $(NNFW_DIR)/include/nnfw

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
NNFW_LIB_PATH := $(NNFW_DIR)/lib
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

#------------------------------------------------------
# nnfw prebuilt shared libraries
#------------------------------------------------------
include $(LOCAL_PATH)/Android-nnfw-prebuilt.mk

#------------------------------------------------------
# tensor-filter sub-plugin for nnfw
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := nnfw-subplugin
LOCAL_SRC_FILES := $(NNSTREAMER_FILTER_NNFW_SRCS)
LOCAL_CFLAGS := -O3 -fPIC
LOCAL_C_INCLUDES := $(NNFW_INCLUDES) $(NNSTREAMER_INCLUDES) $(GST_HEADERS_COMMON)
LOCAL_SHARED_LIBRARIES := $(NNFW_PREBUILT_LIBS)

include $(BUILD_STATIC_LIBRARY)
