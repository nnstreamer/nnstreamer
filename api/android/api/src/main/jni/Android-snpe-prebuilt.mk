#------------------------------------------------------
# SNPE (The Snapdragon Neural Processing Engine)
#
# This mk file defines prebuilt libraries for snpe module.
# (snpe-sdk, arm64-v8a only)
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef SNPE_LIB_PATH
$(error SNPE_LIB_PATH is not defined!)
endif

SNPE_PREBUILT_LIBS :=

#------------------------------------------------------
# snpe-sdk (prebuilt shared library)
#------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := libSNPE
LOCAL_SRC_FILES := $(SNPE_LIB_PATH)/libSNPE.so
include $(PREBUILT_SHARED_LIBRARY)
SNPE_PREBUILT_LIBS += libSNPE

include $(CLEAR_VARS)
LOCAL_MODULE := libsnpe_adsp
LOCAL_SRC_FILES := $(SNPE_LIB_PATH)/libsnpe_adsp.so
include $(PREBUILT_SHARED_LIBRARY)
SNPE_PREBUILT_LIBS += libsnpe_adsp

include $(CLEAR_VARS)
LOCAL_MODULE := libsnpe_dsp_domains
LOCAL_SRC_FILES := $(SNPE_LIB_PATH)/libsnpe_dsp_domains.so
include $(PREBUILT_SHARED_LIBRARY)
SNPE_PREBUILT_LIBS += libsnpe_dsp_domains

include $(CLEAR_VARS)
LOCAL_MODULE := libsnpe_dsp_domains_v2
LOCAL_SRC_FILES := $(SNPE_LIB_PATH)/libsnpe_dsp_domains_v2.so
include $(PREBUILT_SHARED_LIBRARY)
SNPE_PREBUILT_LIBS += libsnpe_dsp_domains_v2

include $(CLEAR_VARS)
LOCAL_MODULE := libhta
LOCAL_SRC_FILES := $(SNPE_LIB_PATH)/libhta.so
include $(PREBUILT_SHARED_LIBRARY)
SNPE_PREBUILT_LIBS += libhta
