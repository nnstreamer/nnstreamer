#------------------------------------------------------
# tensorflow-lite
#
# This mk file defines tensorflow-lite module with prebuilt static library.
# To build and run the example with gstreamer binaries, we built a static library (e.g., libtensorflow-lite.a)
# for Android/Tensorflow-lite from the Tensorflow repository of the Tizen software platform with version 1.9 using the Android-tensorflow-lite.mk.
# - [Tizen] Tensorflow git repository:
#    * Repository: https://review.tizen.org/gerrit/p/platform/upstream/tensorflow
#    * Branch name: sandbox/sangjung/launchpad_bugfix
#    * Commit: 5c3399d [Debian] add dependency on python 2.7, provides of tensorflow
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

TF_LITE_DIR := $(LOCAL_PATH)/tensorflow-lite
TF_LITE_INCLUDES := $(TF_LITE_DIR)/include

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
TF_LITE_LIB_PATH := $(TF_LITE_DIR)/lib/armv7
else ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
TF_LITE_LIB_PATH := $(TF_LITE_DIR)/lib/arm64
else ifeq ($(TARGET_ARCH_ABI),x86)
TF_LITE_LIB_PATH := $(TF_LITE_DIR)/lib/x86
else ifeq ($(TARGET_ARCH_ABI),x86_64)
TF_LITE_LIB_PATH := $(TF_LITE_DIR)/lib/x86_64
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

#------------------------------------------------------
# tensorflow-lite (prebuilt static library)
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := tensorflow-lite-lib
LOCAL_SRC_FILES := $(TF_LITE_LIB_PATH)/libtensorflow-lite.a

include $(PREBUILT_STATIC_LIBRARY)

#------------------------------------------------------
# tensor-filter sub-plugin for tensorflow-lite
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := tensorflow-lite-subplugin
LOCAL_SRC_FILES := $(NNSTREAMER_FILTER_TFLITE_SRCS)
LOCAL_CXXFLAGS := -std=c++11 -O2 -fPIC -frtti -fexceptions
LOCAL_C_INCLUDES := $(TF_LITE_INCLUDES)
LOCAL_STATIC_LIBRARIES := tensorflow-lite-lib cpufeatures
LOCAL_SHARED_LIBRARIES := $(NNS_API_LIBS)

include $(BUILD_SHARED_LIBRARY)
