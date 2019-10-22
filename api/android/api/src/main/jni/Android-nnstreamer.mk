#------------------------------------------------------
# nnstreamer
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := nnstreamer

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

NNSTREAMER_SRC_FILES := \
    $(NNSTREAMER_COMMON_SRCS) \
    $(NNSTREAMER_PLUGINS_SRCS) \
    $(NNSTREAMER_CAPI_SRCS) \
    $(NNSTREAMER_FILTER_TFLITE_SRCS) \
    $(NNSTREAMER_DECODER_BB_SRCS) \
    $(NNSTREAMER_DECODER_DV_SRCS) \
    $(NNSTREAMER_DECODER_IL_SRCS) \
    $(NNSTREAMER_DECODER_PE_SRCS) \
    $(NNSTREAMER_DECODER_IS_SRCS)

# Remove duplicates
LOCAL_SRC_FILES := $(sort $(NNSTREAMER_SRC_FILES))

LOCAL_C_INCLUDES := \
    $(NNSTREAMER_INCLUDES) \
    $(NNSTREAMER_CAPI_INCLUDES)

# common headers (gstreamer, glib)
LOCAL_C_INCLUDES += \
    $(GSTREAMER_ROOT)/include/gstreamer-1.0 \
    $(GSTREAMER_ROOT)/include/glib-2.0 \
    $(GSTREAMER_ROOT)/lib/glib-2.0/include \
    $(GSTREAMER_ROOT)/include

# common headers (tensorflow-lite)
LOCAL_C_INCLUDES += \
    $(TF_LITE_INCLUDES)

LOCAL_CFLAGS += -O2 -DVERSION=\"$(NNSTREAMER_VERSION)\"
LOCAL_CXXFLAGS += -std=c++11 -O2 -DVERSION=\"$(NNSTREAMER_VERSION)\"

include $(BUILD_STATIC_LIBRARY)
