#------------------------------------------------------
# NNFW (On-device neural network inference framework, which is developed by Samsung Research.)
# https://github.com/Samsung/ONE
#
# This mk file defines prebuilt libraries for nnfw module.
# (nnfw core libraries, arm64-v8a only)
# You can download specific version of nnfw libraries from https://github.com/Samsung/ONE/releases.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNFW_LIB_PATH
$(error NNFW_LIB_PATH is not defined!)
endif

NNFW_PREBUILT_LIBS :=

#------------------------------------------------------
# nnfw prebuilt shared libraries
#------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-libnnfw-dev
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libnnfw-dev.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libnnfw-dev
