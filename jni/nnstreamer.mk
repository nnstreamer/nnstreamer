# This mk file defines common features to build NNStreamer library for Android.

ifndef NNSTREAMER_ROOT
$(error NNSTREAMER_ROOT is not defined!)
endif

NNSTREAMER_VERSION  := 0.2.1

NNSTREAMER_GST_HOME := $(NNSTREAMER_ROOT)/gst/nnstreamer
NNSTREAMER_EXT_HOME := $(NNSTREAMER_ROOT)/ext/nnstreamer

# nnstreamer common headers
NNSTREAMER_INCLUDES := \
    $(NNSTREAMER_GST_HOME)

# nnstreamer common sources
NNSTREAMER_COMMON_SRCS := \
    $(NNSTREAMER_GST_HOME)/nnstreamer.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_conf.c \
    $(NNSTREAMER_GST_HOME)/nnstreamer_subplugin.c \
    $(NNSTREAMER_GST_HOME)/tensor_common.c

# nnstreamer plugins
NNSTREAMER_PLUGINS_SRCS := \
    $(NNSTREAMER_GST_HOME)/tensor_converter/tensor_converter.c \
    $(NNSTREAMER_GST_HOME)/tensor_aggregator/tensor_aggregator.c \
    $(NNSTREAMER_GST_HOME)/tensor_decoder/tensordec.c \
    $(NNSTREAMER_GST_HOME)/tensor_demux/gsttensordemux.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter.c \
    $(NNSTREAMER_GST_HOME)/tensor_filter/tensor_filter_custom.c \
    $(NNSTREAMER_GST_HOME)/tensor_merge/gsttensormerge.c \
    $(NNSTREAMER_GST_HOME)/tensor_mux/gsttensormux.c \
    $(NNSTREAMER_GST_HOME)/tensor_repo/tensor_repo.c \
    $(NNSTREAMER_GST_HOME)/tensor_repo/tensor_reposink.c \
    $(NNSTREAMER_GST_HOME)/tensor_repo/tensor_reposrc.c \
    $(NNSTREAMER_GST_HOME)/tensor_sink/tensor_sink.c \
    $(NNSTREAMER_GST_HOME)/tensor_source/tensor_src_iio.c \
    $(NNSTREAMER_GST_HOME)/tensor_split/gsttensorsplit.c \
    $(NNSTREAMER_GST_HOME)/tensor_transform/tensor_transform.c

# filter tensorflow
NNSTREAMER_FILTER_TF_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_tensorflow.c \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_tensorflow_core.cc

# filter tensorflow-lite
NNSTREAMER_FILTER_TFLITE_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_tensorflow_lite.c \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_tensorflow_lite_core.cc

# filter pytorch
NNSTREAMER_FILTER_TORCH_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_pytorch.c \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_pytorch_core.cc

# filter caffe2
NNSTREAMER_FILTER_CAFFE2_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_caffe2.c \
    $(NNSTREAMER_EXT_HOME)/tensor_filter/tensor_filter_caffe2_core.cc

# decoder boundingbox
NNSTREAMER_DECODER_BB_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-boundingbox.c

# decoder directvideo
NNSTREAMER_DECODER_DV_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-directvideo.c

# decoder imagelabel
NNSTREAMER_DECODER_IL_SRCS := \
    $(NNSTREAMER_EXT_HOME)/tensor_decoder/tensordec-imagelabel.c

# common features
NO_AUDIO := false

