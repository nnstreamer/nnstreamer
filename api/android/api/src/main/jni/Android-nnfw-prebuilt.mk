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
LOCAL_MODULE := nnfw-libbackend_cpu
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libbackend_cpu.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libbackend_cpu

include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-libcircle_loader
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libcircle_loader.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libcircle_loader

include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-libneuralnetworks
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libneuralnetworks.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libneuralnetworks

include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-libnnfw-dev
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libnnfw-dev.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libnnfw-dev

#include $(CLEAR_VARS)
#LOCAL_MODULE := nnfw-libnnfw_lib_benchmark
#LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libnnfw_lib_benchmark.so
#include $(PREBUILT_SHARED_LIBRARY)
#NNFW_PREBUILT_LIBS += nnfw-libnnfw_lib_benchmark

include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-libonert_core
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libonert_core.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libonert_core

#include $(CLEAR_VARS)
#LOCAL_MODULE := nnfw-libtensorflowlite_jni
#LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libtensorflowlite_jni.so
#include $(PREBUILT_SHARED_LIBRARY)
#NNFW_PREBUILT_LIBS += nnfw-libtensorflowlite_jni

include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-libtflite_loader
LOCAL_SRC_FILES := $(NNFW_LIB_PATH)/libtflite_loader.so
include $(PREBUILT_SHARED_LIBRARY)
NNFW_PREBUILT_LIBS += nnfw-libtflite_loader
