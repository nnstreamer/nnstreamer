LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifndef NNSTREAMER_ROOT
NNSTREAMER_ROOT := $(LOCAL_PATH)/..
endif

ifndef GSTREAMER_ROOT_ANDROID
$(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

ifeq ($(TARGET_ARCH_ABI),armeabi)
GSTREAMER_ROOT        := $(GSTREAMER_ROOT_ANDROID)/arm
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
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

# Common definition for NNStreamer
include $(LOCAL_PATH)/nnstreamer.mk

# Define shared libraries that are required by a gstreamer plug-in.
define shared_lib_common
    include $(CLEAR_VARS)
    LOCAL_MODULE := $(1)
    LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/lib$(1).so
    $(info $(GSTREAMER_ROOT)/lib/lib$(1).so)
    include $(PREBUILT_SHARED_LIBRARY)
endef

# Define shared libraries that are used as a gstreamer plug-in.
define shared_lib_gst
    include $(CLEAR_VARS)
    LOCAL_MODULE := $(1)
    LOCAL_SRC_FILES := $(GSTREAMER_ROOT)/lib/gstreamer-1.0/lib$(1).so
    $(info $(GSTREAMER_ROOT)/lib/gstreamer-1.0/lib$(1).so)
    include $(PREBUILT_SHARED_LIBRARY)
endef

$(foreach item,$(GST_LIBS_COMMON),$(eval $(call shared_lib_common,$(item))))

$(foreach item,$(GST_LIBS_GST),$(eval $(call shared_lib_gst,$(item))))

include $(CLEAR_VARS)

# Please keep the pthread and openmp library for checking a compatibility
LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -fPIC -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -std=c++11 -fPIC -frtti -fexceptions -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

ifeq ($(NO_AUDIO), true)
LOCAL_CFLAGS += -DNO_AUDIO
LOCAL_CXXFLAGS += -DNO_AUDIO
endif

LOCAL_LDFLAGS       += -fuse-ld=bfd
LOCAL_MODULE_TAGS   := optional

LOCAL_MODULE        := nnstreamer
LOCAL_SRC_FILES     := $(NNSTREAMER_COMMON_SRCS) $(NNSTREAMER_PLUGINS_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

LOCAL_SHARED_LIBRARIES := $(GST_BUILDING_BLOCK_LIST)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)

ifndef TENSORFLOW_ROOT
ifneq ($(MAKECMDGOALS),clean)
$(warning TENSORFLOW_ROOT is not defined!)
$(warning TENSORFLOW SRC is going to be downloaded!)

# Currently we are using tensorflow 1.9.0
$(info $(shell ($(LOCAL_PATH)/prepare_tflite.sh)))

TENSORFLOW_ROOT := $(LOCAL_PATH)/tensorflow-1.9.0

endif
endif

TF_LITE_DIR=$(TENSORFLOW_ROOT)/tensorflow/contrib/lite

LOCAL_MODULE := tensorflow-lite
TFLITE_SRCS := \
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

TFLITE_SRCS := $(sort $(TFLITE_SRCS))

TFLITE_EXCLUDE_SRCS := \
    $(wildcard $(TF_LITE_DIR)/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/*/*/*/*test.cc) \
    $(wildcard $(TF_LITE_DIR)/kernels/test_util.cc) \
    $(wildcard $(TF_LITE_DIR)/examples/minimal/minimal.cc)

TFLITE_SRCS := $(filter-out $(TFLITE_EXCLUDE_SRCS), $(TFLITE_SRCS))
# ANDROID_NDK env should be set before build
TFLITE_INCLUDES := \
    $(ANDROID_NDK)/../ \
    $(TENSORFLOW_ROOT) \
    $(TF_LITE_DIR)/downloads \
    $(TF_LITE_DIR)/downloads/eigen \
    $(TF_LITE_DIR)/downloads/gemmlowp \
    $(TF_LITE_DIR)/downloads/neon_2_sse \
    $(TF_LITE_DIR)/downloads/farmhash/src \
    $(TF_LITE_DIR)/downloads/flatbuffers/include


LOCAL_SRC_FILES := $(TFLITE_SRCS)
LOCAL_C_INCLUDES := $(TFLITE_INCLUDES)

LOCAL_CFLAGS += -O3 -DNDEBUG
LOCAL_CXXFLAGS += -std=c++11 -frtti -fexceptions -O3 -DNDEBUG

include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnstreamer_filter_tensorflow-lite
LOCAL_SRC_FILES := $(NNSTREAMER_FILTER_TFLITE_SRCS)
LOCAL_C_INCLUDES  := $(NNSTREAMER_INCLUDES) $(TFLITE_INCLUDES)

LOCAL_SHARED_LIBRARIES := $(GST_BUILDING_BLOCK_LIST) nnstreamer
LOCAL_STATIC_LIBRARIES := tensorflow-lite cpufeatures

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnstreamer_decoder_bounding_boxes
LOCAL_SRC_FILES := $(NNSTREAMER_DECODER_BB_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

LOCAL_SHARED_LIBRARIES := $(GST_BUILDING_BLOCK_LIST) nnstreamer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -std=c++11 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnstreamer_decoder_image_labeling
LOCAL_SRC_FILES := $(NNSTREAMER_DECODER_IL_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

LOCAL_SHARED_LIBRARIES := $(GST_BUILDING_BLOCK_LIST) nnstreamer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -std=c++11 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnstreamer_decoder_direct_video
LOCAL_SRC_FILES := $(NNSTREAMER_DECODER_DV_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

LOCAL_SHARED_LIBRARIES := $(GST_BUILDING_BLOCK_LIST) nnstreamer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -std=c++11 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nnstreamer_decoder_pose_estimation
LOCAL_SRC_FILES := $(NNSTREAMER_DECODER_PE_SRCS)
LOCAL_C_INCLUDES    := $(NNSTREAMER_INCLUDES)

LOCAL_SHARED_LIBRARIES := $(GST_BUILDING_BLOCK_LIST) nnstreamer

LOCAL_ARM_NEON      := true
LOCAL_CFLAGS        += -O0 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS      += -std=c++11 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CFLAGS        += -pthread -fopenmp

LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

include $(BUILD_SHARED_LIBRARY)

$(call import-module, android/cpufeatures)
