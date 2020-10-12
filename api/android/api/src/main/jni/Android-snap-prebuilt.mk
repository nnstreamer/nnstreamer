#------------------------------------------------------
# SNAP (Samsung Neural Acceleration Platform)
#
# This mk file defines prebuilt libraries for snap module.
# (snap-sdk, arm64-v8a only)
#------------------------------------------------------
LOCAL_PATH := $(call my-dir)

ifndef SNAP_LIB_PATH
$(error SNAP_LIB_PATH is not defined!)
endif

SNAP_PREBUILT_LIBS :=

#------------------------------------------------------
# snap-sdk (prebuilt shared library)
#------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := snap-sdk
LOCAL_SRC_FILES := $(SNAP_LIB_PATH)/libsnap_vndk.so
include $(PREBUILT_SHARED_LIBRARY)
SNAP_PREBUILT_LIBS += snap-sdk
