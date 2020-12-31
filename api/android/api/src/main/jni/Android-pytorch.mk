#------------------------------------------------------
# PyTorch
#
# This mk file defines PyTorch module with prebuilt static library.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

# To support NNAPI, which is not available in the lastest stable release (1.7.1) of PyTorch,
# This module use commit ID 5c3788d5d76f64f6708e0b79f40b1cf45276625a for PyTorch
# (https://github.com/pytorch/pytorch @ 5c3788d5d76f64f6708e0b79f40b1cf45276625a)
# After a release of PyTorch which includes NNAPI support, this will be updated.
PYTORCH_VERSION := 1.8.0

PYTORCH_FLAGS := \
    -DPYTORCH_VERSION=$(PYTORCH_VERSION) \
    -DPYTORCH_VER_ATLEAST_1_2_0=1

PYTORCH_DIR := $(LOCAL_PATH)/pytorch
PYTORCH_INCLUDES := $(PYTORCH_DIR)/include

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
PYTORCH_LIB_PATH := $(PYTORCH_DIR)/lib/arm64
else
$(error Target arch ABI not supported: $(TARGET_ARCH_ABI))
endif

#------------------------------------------------------
# pytorch (prebuilt static library)
#------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libc10
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libc10.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libclog
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libclog.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libcpuinfo
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libcpuinfo.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libeigen_blas
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libeigen_blas.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libnnpack
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libnnpack.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libpthreadpool
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libpthreadpool.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libpytorch_qnnpack
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libpytorch_qnnpack.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libXNNPACK
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libXNNPACK.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libtorch_cpu
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libtorch_cpu.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := pytorch-libtorch
LOCAL_SRC_FILES := $(PYTORCH_LIB_PATH)/libtorch.a
include $(PREBUILT_STATIC_LIBRARY)

#------------------------------------------------------
# tensor-filter sub-plugin for pytorch
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := pytorch-subplugin
LOCAL_SRC_FILES := $(NNSTREAMER_FILTER_PYTORCH_SRCS)
LOCAL_CXXFLAGS := -std=c++14 -O3 -fPIC -frtti -fexceptions $(NNS_API_FLAGS) $(PYTORCH_FLAGS)
LOCAL_C_INCLUDES := $(PYTORCH_INCLUDES) $(NNSTREAMER_INCLUDES) $(GST_HEADERS_COMMON)
LOCAL_WHOLE_STATIC_LIBRARIES := pytorch-libeigen_blas pytorch-libnnpack pytorch-libpytorch_qnnpack pytorch-libXNNPACK pytorch-libtorch_cpu pytorch-libtorch pytorch-libc10 pytorch-libcpuinfo pytorch-libpthreadpool pytorch-libclog
include $(BUILD_STATIC_LIBRARY)
