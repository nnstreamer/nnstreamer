/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-Decoder
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	tensor_decoder_custom.h
 * @date	22 Mar 2021
 * @brief	NNStreamer APIs for tensor_decoder custom condition
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __NNS_TENSOR_DECODER_CUSTOM_H__
#define __NNS_TENSOR_DECODER_CUSTOM_H__

#include <glib.h>
#include <gst/gst.h>
#include "tensor_typedef.h"

G_BEGIN_DECLS
/**
 * @brief Decode from tensors to media as customized operation
 * @param[in] input the input memory containg tensors
 * @param[in] config input tensors config
 * @param[in] data private data for the callback
 * @param[out] output buffer filled by user
 * @return 0 if success. -ERRNO if error.
 */
typedef int (* tensor_decoder_custom) (const GstTensorMemory *input,
    const GstTensorsConfig *config, void *data, GstBuffer * out_buf);

/**
 * @brief Register the custom callback function.
 * @param[in] name The name of tensor_decoder custom callback function.
 * @param[in] func The custom condition function body
 * @param[in/out] data The internal data for the function
 * @return 0 if success. -ERRNO if error.
 */
extern int
nnstreamer_decoder_custom_register (const gchar *name, tensor_decoder_custom func, void *data);

/**
 * @brief Unregister the custom callback function.
 * @param[in] name The registered name of tensor_decoder custom callback function.
 * @return 0 if success. -ERRNO if error.
 */
extern int
nnstreamer_decoder_custom_unregister (const gchar *name);

G_END_DECLS
#endif /*__NNS_TENSOR_DECODER_CUSTOM_H__*/
