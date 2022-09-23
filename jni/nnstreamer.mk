# This mk file defines common features to build NNStreamer library for Android.

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

ifndef GSTREAMER_ROOT_ANDROID
$(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

NNSTREAMER_VERSION  := 2.3.0
NNSTREAMER_VERSION_MAJOR := $(word 1,$(subst ., ,${NNSTREAMER_VERSION}))
NNSTREAMER_VERSION_MINOR := $(word 2,$(subst ., ,${NNSTREAMER_VERSION}))
NNSTREAMER_VERSION_MICRO := $(word 3,$(subst ., ,${NNSTREAMER_VERSION}))

NNSTREAMER_GST_HOME := $(NNSTREAMER_ROOT)/gst/nnstreamer
NNSTREAMER_EXT_HOME := $(NNSTREAMER_ROOT)/ext/nnstreamer

CMDRESULT1 := $(shell sed "s/@__NNSTREAMER_VERSION_MAJOR__@/${NNSTREAMER_VERSION_MAJOR}/" ${NNSTREAMER_GST_HOME}/include/nnstreamer_version.h.in > ${NNSTREAMER_GST_HOME}/include/nnstreamer_version.h && echo "Processed sed 1")
CMDRESULT2 := $(shell sed -i "s/@__NNSTREAMER_VERSION_MINOR__@/${NNSTREAMER_VERSION_MINOR}/" ${NNSTREAMER_GST_HOME}/include/nnstreamer_version.h && echo "Processed sed 2")
CMDRESULT3 := $(shell sed -i "s/@__NNSTREAMER_VERSION_MICRO__@/${NNSTREAMER_VERSION_MICRO}/" ${NNSTREAMER_GST_HOME}/include/nnstreamer_version.h && echo "Processed sed 3")

$(info ${CMDRESULT1})
$(info ${CMDRESULT2})
$(info ${CMDRESULT3})

# nnstreamer common headers
NNSTREAMER_INCLUDES := \
    $(NNSTREAMER_GST_HOME) \
    $(NNSTREAMER_GST_HOME)/include

# nnstreamer common sources. (including tensor-filter common, custom filter)
NNSTREAMER_COMMON_SRCS := \
    $(NNSTREAMER_GST_HOME)/hw_accel.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_conf.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_log.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_subplugin.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_plugin_api_util_impl.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_common.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_custom.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_custom_easy.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_support_cc.cc \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_single.c \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_cpp.cc

# nnstreamer plugins. Not used for SINGLE-only build.
NNSTREAMER_PLUGINS_SRCS := \
    $(NNSTREAMER_GST_HOME)/tensor_data.c \
    $(NNSTREAMER_GST_HOME)/tensor_meta.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_plugin_api_impl.c \
    $(NNSTREAMER_GST_HOME)/registerer/nnstreamer.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_aggregator.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_converter.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_crop.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_debug.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_decoder.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_demux.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_if.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_merge.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_mux.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_rate.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_repo.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_reposink.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_reposrc.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_sink.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_sparsedec.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_sparseenc.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_sparseutil.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_split.c \
    $(NNSTREAMER_GST_HOME)/elements/gsttensor_transform.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter.c

# tensor-query element with nnstreamer-edge
NNSTREAMER_QUERY_SRCS := \
    $(NNSTREAMER_GST_HOME)/tensor_query/tensor_query_common.c \
    $(NNSTREAMER_GST_HOME)/tensor_query/tensor_query_client.c \
    $(NNSTREAMER_GST_HOME)/tensor_query/tensor_query_serversink.c \
    $(NNSTREAMER_GST_HOME)/tensor_query/tensor_query_serversrc.c \
    $(NNSTREAMER_GST_HOME)/tensor_query/tensor_query_server.c

# source AMC (Android MediaCodec)
NNSTREAMER_SOURCE_AMC_SRCS := \
    $(NNSTREAMER_EXT_HOME)/android_source/gstamcsrc.c \
    $(NNSTREAMER_EXT_HOME)/android_source/gstamcsrc_looper.cc

# filter tensorflow
NNSTREAMER_FILTER_TF_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_tensorflow.cc

# filter tensorflow-lite
NNSTREAMER_FILTER_TFLITE_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_tensorflow_lite.cc

# filter nnfw
NNSTREAMER_FILTER_NNFW_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_nnfw.c

# filter pytorch
## NOTE: ndk does not support PYTORCH; it requires gcc.
NNSTREAMER_FILTER_PYTORCH_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_pytorch.cc

# filter caffe2
NNSTREAMER_FILTER_CAFFE2_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_caffe2.cc

# filter snpe
NNSTREAMER_FILTER_SNPE_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_snpe.cc

# filter snap
NNSTREAMER_FILTER_SNAP_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_snap.cc

# filter mxnet
NNSTREAMER_FILTER_MXNET_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_mxnet.cc

# converter flatbuffers
NNSTREAMER_CONVERTER_FLATBUF_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_converter/tensor_converter_util.c \
    $(NNSTREAMER_EXT_HOME)/tensor_converter/tensor_converter_flatbuf.cc

# converter flexbuffers
NNSTREAMER_CONVERTER_FLEXBUF_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_converter/tensor_converter_util.c \
    $(NNSTREAMER_EXT_HOME)/tensor_converter/tensor_converter_flexbuf.cc

# decoder flatbuffers
NNSTREAMER_DECODER_FLATBUF_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-flatbuf.cc \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c

# decoder flexbuffers
NNSTREAMER_DECODER_FLEXBUF_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-flexbuf.cc \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c

# decoder boundingbox
NNSTREAMER_DECODER_BB_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-boundingbox.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-font.c

# decoder directvideo
NNSTREAMER_DECODER_DV_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-directvideo.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c

# decoder imagelabel
NNSTREAMER_DECODER_IL_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-imagelabel.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c

# decoder pose estimation
NNSTREAMER_DECODER_PE_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-pose.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-font.c

# decoder image segment
NNSTREAMER_DECODER_IS_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-imagesegment.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c

# decoder octet-stream
NNSTREAMER_DECODER_OS_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-octetstream.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c

# gstreamer join element
NNSTREAMER_JOIN_SRCS := \
    $(NNSTREAMER_ROOT)/gst/join/gstjoin.c

# gstreamer mqtt element
NNSTREAMER_MQTT_SRCS := \
    $(NNSTREAMER_ROOT)/gst/mqtt/mqttelements.c \
    $(NNSTREAMER_ROOT)/gst/mqtt/mqttsink.c \
    $(NNSTREAMER_ROOT)/gst/mqtt/mqttsrc.c \
    $(NNSTREAMER_ROOT)/gst/mqtt/ntputil.c

# common features
NO_AUDIO := false

GST_HEADERS_COMMON := \
    $(GSTREAMER_ROOT)/include/gstreamer-1.0 \
    $(GSTREAMER_ROOT)/include/glib-2.0 \
    $(GSTREAMER_ROOT)/include/json-glib-1.0 \
    $(GSTREAMER_ROOT)/lib/glib-2.0/include \
    $(GSTREAMER_ROOT)/include

GST_LIBS_COMMON := gstreamer-1.0 gstbase-1.0 gstvideo-1.0 glib-2.0 \
                   gobject-2.0 intl z bz2 orc-0.4 gmodule-2.0 gsttag-1.0 iconv \
                   gstapp-1.0 png16 gio-2.0 pangocairo-1.0 \
                   pangoft2-1.0 pango-1.0 gthread-2.0 cairo pixman-1 fontconfig expat freetype \
                   gstcontroller-1.0 jpeg graphene-1.0 gstpbutils-1.0 gstgl-1.0 \
                   gstallocators-1.0 harfbuzz gstphotography-1.0 ffi fribidi gstnet-1.0 \
		   cairo-gobject cairo-script-interpreter

ifeq ($(NO_AUDIO), false)
GST_LIBS_COMMON += gstaudio-1.0 gstbadaudio-1.0
endif

GST_LIBS_GST := gstcoreelements gstcoretracers gstadder gstapp \
                gstpango gstrawparse gsttypefindfunctions gstvideoconvert gstvideorate \
                gstvideoscale gstvideotestsrc gstvolume gstautodetect gstvideofilter gstvideocrop gstopengl \
                gstopensles gstcompositor gstpng gstmultifile gstvideomixer gsttcp gstjpegformat gstcairo

ifeq ($(NO_AUDIO), false)
GST_LIBS_GST += gstaudioconvert gstaudiomixer gstaudiorate gstaudioresample gstaudiotestsrc gstjpeg
endif

# gstreamer building block for nstreamer
GST_BUILDING_BLOCK_LIST := $(GST_LIBS_COMMON) $(GST_LIBS_GST)

# gstreamer building block for decoder and filter
NNSTREAMER_BUILDING_BLOCK_LIST := $(GST_BUILDING_BLOCK_LIST) nnstreamer nnstreamer_decoder_bounding_boxes nnstreamer_decoder_pose_estimation nnstreamer_filter_tensorflow-lite nnstreamer_decoder_flatbuf
