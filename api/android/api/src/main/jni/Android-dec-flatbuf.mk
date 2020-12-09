#------------------------------------------------------
# flatbuffers
#
# This mk file defines the flatbuffers-module with the prebuilt static library.
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

include $(NNSTREAMER_ROOT)/jni/nnstreamer.mk

FLATBUF_VER := @FLATBUF_VER@
ifeq ($(FLATBUF_VER),@FLATBUF_VER@)
$(error 'FLATBUF_VER' is not properly set)
endif

ifeq ($(shell which flatc),)
$(error No 'flatc' in your PATH, install flatbuffers-compiler from ppa:nnstreamer/ppa)
else
SYS_FLATC_VER := $(word 3, $(shell flatc --version))
endif

ifneq ($(SYS_FLATC_VER), $(FLATBUF_VER))
$(error Found 'flatc' v$(SYS_FLATC_VER), but required v$(FLATBUF_VER))
endif

FLATBUF_DIR := $(LOCAL_PATH)/flatbuffers
FLATBUF_INCLUDES := $(FLATBUF_DIR)/include
GEN_FLATBUF_HEADER := $(shell flatc --cpp -o $(LOCAL_PATH) $(NNSTREAMER_ROOT)/ext/nnstreamer/include/nnstreamer.fbs )
FLATBUF_HEADER_GEN := $(wildcard $(LOCAL_PATH)/nnstreamer_generated.h)
ifeq ($(FLATBUF_HEADER_GEN), '')
$(error Failed to generate the header file, '$(LOCAL_PATH)/nnstreamer_generated.h')
endif

FLATBUF_LIB_PATH := $(FLATBUF_DIR)/lib/$(TARGET_ARCH_ABI)
ifeq ($(wildcard $(FLATBUF_LIB_PATH)), )
$(error The given ABI is not supported by the flatbuffers-module: $(TARGET_ARCH_ABI))
endif

#------------------------------------------------------
# libflatbuffers.a (prebuilt static library)
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := flatbuffers-lib
LOCAL_SRC_FILES := $(FLATBUF_LIB_PATH)/libflatbuffers.a

include $(PREBUILT_STATIC_LIBRARY)

#------------------------------------------------------
# tensor-decoder sub-plugin for flatbuffers
#------------------------------------------------------
include $(CLEAR_VARS)

LOCAL_MODULE := flatbuffers-subplugin
LOCAL_SRC_FILES := $(NNSTREAMER_DECODER_FLATBUF_SRCS)
LOCAL_C_INCLUDES := $(LOCAL_PATH) $(FLATBUF_INCLUDES) $(NNSTREAMER_INCLUDES) $(GST_HEADERS_COMMON)
LOCAL_STATIC_LIBRARIES := flatbuffers-lib

include $(BUILD_STATIC_LIBRARY)
