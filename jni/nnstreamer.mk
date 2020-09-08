# This mk file defines common features to build NNStreamer library for Android.

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

ifndef GSTREAMER_ROOT_ANDROID
$(error GSTREAMER_ROOT_ANDROID is not defined!)
endif

NNSTREAMER_VERSION  := 1.5.3
NNSTREAMER_VERSION_MAJOR := $(word 1,$(subst ., ,${NNSTREAMER_VERSION}))
NNSTREAMER_VERSION_MINOR := $(word 2,$(subst ., ,${NNSTREAMER_VERSION}))
NNSTREAMER_VERSION_MICRO := $(word 3,$(subst ., ,${NNSTREAMER_VERSION}))

NNSTREAMER_GST_HOME := $(NNSTREAMER_ROOT)/gst/nnstreamer
NNSTREAMER_EXT_HOME := $(NNSTREAMER_ROOT)/ext/nnstreamer
NNSTREAMER_CAPI_HOME := $(NNSTREAMER_ROOT)/api/capi

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
    $(NNSTREAMER_GST_HOME)/nnstreamer_subplugin.c \
    $(NNSTREAMER_GST_HOME)/tensor_common.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_common.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_custom.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_custom_easy.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_support_cc.cc \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_cpp.cc

# nnstreamer plugins. Not used for SINGLE-only build.
NNSTREAMER_PLUGINS_SRCS := \
    $(NNSTREAMER_GST_HOME)/tensor_common_pipeline.c \
    $(NNSTREAMER_GST_HOME)/registerer/nnstreamer.c \
    $(NNSTREAMER_GST_HOME)/tensor_converter/tensor_converter.c \
    $(NNSTREAMER_GST_HOME)/tensor_aggregator/tensor_aggregator.c \
    $(NNSTREAMER_GST_HOME)/tensor_decoder/tensordec.c \
    $(NNSTREAMER_GST_HOME)/tensor_demux/gsttensordemux.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter.c \
    $(NNSTREAMER_GST_HOME)/tensor_merge/gsttensormerge.c \
    $(NNSTREAMER_GST_HOME)/tensor_mux/gsttensormux.c \
    $(NNSTREAMER_GST_HOME)/tensor_repo/tensor_repo.c \
    $(NNSTREAMER_GST_HOME)/tensor_repo/tensor_reposink.c \
    $(NNSTREAMER_GST_HOME)/tensor_repo/tensor_reposrc.c \
    $(NNSTREAMER_GST_HOME)/tensor_sink/tensor_sink.c \
    $(NNSTREAMER_GST_HOME)/tensor_split/gsttensorsplit.c \
    $(NNSTREAMER_GST_HOME)/tensor_transform/tensor_transform.c \
    $(NNSTREAMER_GST_HOME)/tensor_if/gsttensorif.c

# nnstreamer c-api
NNSTREAMER_CAPI_INCLUDES := \
    $(NNSTREAMER_ROOT)/gst \
    $(NNSTREAMER_CAPI_HOME)/include/platform \
    $(NNSTREAMER_CAPI_HOME)/include

# nnstreamer c-api (single+pipeline). requires NNSTREAMER_PLUGINS_SRCS as well.
NNSTREAMER_CAPI_SRCS := \
    $(NNSTREAMER_CAPI_HOME)/src/nnstreamer-capi-pipeline.c \
    $(NNSTREAMER_CAPI_HOME)/src/nnstreamer-capi-single.c \
    $(NNSTREAMER_CAPI_HOME)/src/nnstreamer-capi-util.c \
    $(NNSTREAMER_CAPI_HOME)/src/tensor_filter_single.c

# nnstreamer c-api for single-shot only
NNSTREAMER_SINGLE_SRCS := \
    $(NNSTREAMER_CAPI_HOME)/src/nnstreamer-capi-single.c \
    $(NNSTREAMER_CAPI_HOME)/src/nnstreamer-capi-util.c \
    $(NNSTREAMER_CAPI_HOME)/src/tensor_filter_single.c

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
NNSTREAMER_FILTER_TORCH_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_pytorch.cc

# filter caffe2
NNSTREAMER_FILTER_CAFFE2_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_caffe2.cc

# filter snpe
NNSTREAMER_FILTER_SNPE_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_snpe.cc

# decoder boundingbox
NNSTREAMER_DECODER_BB_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-boundingbox.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-font.c

# decoder directvideo
NNSTREAMER_DECODER_DV_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-directvideo.c

# decoder imagelabel
NNSTREAMER_DECODER_IL_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-imagelabel.c

# decoder pose estimation
NNSTREAMER_DECODER_PE_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-pose.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordecutil.c \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-font.c

# decoder image segment
NNSTREAMER_DECODER_IS_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-imagesegment.c

# common features
NO_AUDIO := false

ENABLE_NNAPI :=false

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
NNSTREAMER_BUILDING_BLOCK_LIST := $(GST_BUILDING_BLOCK_LIST) nnstreamer nnstreamer_decoder_bounding_boxes nnstreamer_decoder_pose_estimation nnstreamer_filter_tensorflow-lite

# libs for nnapi
NNAPI_BUILDING_BLOCK := arm_compute_ex backend_acl_cl backend_acl_neon backend_cpu \
                        neuralnetworks arm_compute_core arm_compute_graph arm_compute OpenCL

ifeq ($(ENABLE_NNAPI), true)
NNSTREAMER_BUILDING_BLOCK_LIST += $(NNAPI_BUILDING_BLOCK)
endif
