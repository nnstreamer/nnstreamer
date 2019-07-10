#------------------------------------------------------
# tensorflow-lite
#
# This mk file is to build a static library from cloned tensorflow repository.
# To utilize and build a new version, you have to define the root directory and check Makefile to build tensorflow-lite.
# Now this file is to build tensorflow-lite from tizen tensorflow repository with version 1.9.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := tensorflow-lite

# Need to set tensorflow root dir
ifndef TENSORFLOW_ROOT
$(error TENSORFLOW_ROOT is not defined!)
endif

# Set tensorflow-lite dir (TODO change this with tensorflow-lite version)
TF_LITE_DIR := $(TENSORFLOW_ROOT)/tensorflow/contrib/lite

# Set files to compile (TODO check Makefile to build tensorflow-lite)
CORE_CC_ALL_SRCS := \
    $(wildcard $(TF_LITE_DIR)/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/optimized/*.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/reference/*.cc) \
    $(wildcard $(TF_LITE_DIR)/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/optimized/*.c) \
    $(wildcard $(TF_LITE_DIR)/kernels/internal/reference/*.c) \
    $(wildcard $(TF_LITE_DIR)/downloads/farmhash/src/farmhash.cc) \
    $(wildcard $(TF_LITE_DIR)/downloads/fft2d/fftsg.c)

# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
    $(wildcard $(TF_LITE_DIR)/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/test_util.cc) \
    $(wildcard $(TF_LITE_DIR)/examples/minimal/minimal.cc)

# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))
TF_LITE_INCLUDES := \
    $(ANDROID_NDK)/../ \
    $(TENSORFLOW_ROOT) \
    $(TF_LITE_DIR)/downloads \
    $(TF_LITE_DIR)/downloads/eigen \
    $(TF_LITE_DIR)/downloads/gemmlowp \
    $(TF_LITE_DIR)/downloads/neon_2_sse \
    $(TF_LITE_DIR)/downloads/farmhash/src \
    $(TF_LITE_DIR)/downloads/flatbuffers/include

LOCAL_SRC_FILES := $(TF_LITE_CC_SRCS)
LOCAL_C_INCLUDES := $(TF_LITE_INCLUDES)

LOCAL_CFLAGS += -O2 -DNDEBUG
# std for toolchain in NDK
# rtti for typecast in tensorflow-lite
# exceptions to enable exception handling in tensorflow-lite
LOCAL_CXXFLAGS += -std=c++11 -frtti -fexceptions -O2 -DNDEBUG

include $(BUILD_STATIC_LIBRARY)
