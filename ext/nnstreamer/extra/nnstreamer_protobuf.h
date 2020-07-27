/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer subplugin "protobuf" to support tensor converter and decoder
 * Copyright (C) 2020 Gichan Jang <gichan2.jang@samsung.com>
 */
 /**
 * @file        nnstreamer_protobuf.h
 * @date        16 June 2020
 * @brief       Protobuf util function for nnstreamer
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Gichan Jang <gichan2.jang@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#ifndef __NNS_PROTOBUF_UTIL_H__
#define __NNS_PROTOBUF_UTIL_H__

#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>

/**
 * @brief Default static capability for Protocol Buffers
 * protobuf converter will convert this capability to other/tensor(s)
 */
#define GST_PROTOBUF_TENSOR_CAP_DEFAULT \
    "other/protobuf-tensor, " \
    "framerate = " GST_TENSOR_RATE_RANGE

/**
 * @brief tensordec-plugin's GstTensorDecoderDef callback
 * @param[in] config The structure of input tensor info.
 * @param[in] input The array of input tensor data. The maximum array size of input data is NNS_TENSOR_SIZE_LIMIT.
 * @param[out] outbuf A sub-plugin should update or append proper memory for the negotiated media type.
 * @return GST_FLOW_OK if OK.
 */
GstFlowReturn
gst_tensor_decoder_protobuf (const GstTensorsConfig * config,
    const GstTensorMemory * input, GstBuffer * outbuf);

/**
 * @brief tensor converter plugin's NNStreamerExternalConverter callback 
 * @param[in] buf The input stream buffer
 * @param[out] frame_size The size of each frame (output buffer)
 * @param[out] frames_in The number of frames in the given input buffer.
 * @param[out] config tensors config structure to be filled
 * @retval Return input buffer(in_buf) if the data is to be kept untouched.
 * @retval Return a new GstBuf if the data is to be modified.
 */
GstBuffer *
gst_tensor_converter_protobuf (GstBuffer * in_buf, gsize * frame_size,
    guint * frames_in, GstTensorsConfig * config);

#endif /* __NNS_PROTOBUF_UTIL_H__ */
