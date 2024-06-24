#------------------------------------------------------
# Define GStreamer plugins and extra dependencies
#------------------------------------------------------

ifndef GSTREAMER_NDK_BUILD_PATH
GSTREAMER_NDK_BUILD_PATH := $(GSTREAMER_ROOT)/share/gst-android/ndk-build
endif

include $(GSTREAMER_NDK_BUILD_PATH)/plugins.mk

GST_REQUIRED_PLUGINS := $(GSTREAMER_PLUGINS_CORE) \
    $(GSTREAMER_PLUGINS_CODECS) \
    $(GSTREAMER_PLUGINS_ENCODING) \
    $(GSTREAMER_PLUGINS_NET) \
    $(GSTREAMER_PLUGINS_PLAYBACK) \
    $(GSTREAMER_PLUGINS_VIS) \
    $(GSTREAMER_PLUGINS_SYS) \
    $(GSTREAMER_PLUGINS_EFFECTS) \
    $(GSTREAMER_PLUGINS_CAPTURE) \
    $(GSTREAMER_PLUGINS_CODECS_GPL) \
    $(GSTREAMER_PLUGINS_CODECS_RESTRICTED) \
    $(GSTREAMER_PLUGINS_NET_RESTRICTED) \
    $(GSTREAMER_PLUGINS_GES)
GST_REQUIRED_DEPS := gstreamer-video-1.0 gstreamer-audio-1.0 gstreamer-app-1.0
GSTREAMER_EXTRA_LIBS     := $(GST_REQUIRED_LIBS) -liconv

ifeq ($(NO_AUDIO), false)
GST_REQUIRED_DEPS += gstaudio-1.0 gstbadaudio-1.0
endif
