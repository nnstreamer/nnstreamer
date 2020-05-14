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

# common headers (nnstreamer, gstreamer, glib)
NNSTREAMER_INC := \
    $(NNSTREAMER_INCLUDES) \
    $(NNSTREAMER_CAPI_INCLUDES) \
    $(GST_HEADERS_COMMON)

include $(CLEAR_VARS)

LOCAL_MODULE := nnstreamer

LOCAL_SRC_FILES := $(sort $(NNSTREAMER_SRC_FILES))
LOCAL_C_INCLUDES := $(NNSTREAMER_INC)
LOCAL_CFLAGS := -O2 -fPIC $(NNS_API_FLAGS)
LOCAL_CXXFLAGS := -std=c++11 -O2 -fPIC -frtti -fexceptions $(NNS_API_FLAGS)
LOCAL_SHARED_LIBRARIES := gstreamer_android
LOCAL_LDLIBS := -llog -landroid -lmediandk

LOCAL_EXPORT_C_INCLUDES := $(NNSTREAMER_INC)
LOCAL_EXPORT_CFLAGS := $(NNS_API_FLAGS)
LOCAL_EXPORT_CXXFLAGS := $(NNS_API_FLAGS)
LOCAL_EXPORT_LDLIBS := -llog -landroid

include $(BUILD_SHARED_LIBRARY)
