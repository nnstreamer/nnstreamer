#------------------------------------------------------
# nnstreamer
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

NNSTREAMER_SRC_FILES := \
    $(NNSTREAMER_COMMON_SRCS)

ifeq ($(NNSTREAMER_API_OPTION),single)
# single-shot only
NNSTREAMER_SRC_FILES += \
    $(NNSTREAMER_SINGLE_SRCS)
else
# capi and nnstreamer plugins
NNSTREAMER_SRC_FILES += \
    $(NNSTREAMER_CAPI_SRCS) \
    $(NNSTREAMER_PLUGINS_SRCS) \
    $(NNSTREAMER_SOURCE_AMC_SRCS) \
    $(NNSTREAMER_FILTER_CPP_SRCS) \
    $(NNSTREAMER_DECODER_BB_SRCS) \
    $(NNSTREAMER_DECODER_DV_SRCS) \
    $(NNSTREAMER_DECODER_IL_SRCS) \
    $(NNSTREAMER_DECODER_PE_SRCS) \
    $(NNSTREAMER_DECODER_IS_SRCS)
endif

include $(CLEAR_VARS)

LOCAL_MODULE := nnstreamer

# Remove duplicates
LOCAL_SRC_FILES := $(sort $(NNSTREAMER_SRC_FILES))

LOCAL_C_INCLUDES := \
    $(NNSTREAMER_INCLUDES) \
    $(NNSTREAMER_CAPI_INCLUDES)

# common headers (gstreamer, glib)
LOCAL_C_INCLUDES += $(GST_HEADERS_COMMON)

LOCAL_CFLAGS += -O2 $(NNS_API_FLAGS)
LOCAL_CXXFLAGS += -std=c++11 -O2 $(NNS_API_FLAGS)

include $(BUILD_STATIC_LIBRARY)
