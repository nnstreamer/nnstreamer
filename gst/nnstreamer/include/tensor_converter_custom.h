/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-Converter
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	tensor_converter_custom.h
 * @date	18 Mar 2021
 * @brief	NNStreamer APIs for tensor_converter custom condition
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __NNS_TENSOR_CONVERTER_CUSTOM_H__
#define __NNS_TENSOR_CONVERTER_CUSTOM_H__

#include <glib.h>
#include <gst/gst.h>
#include "tensor_typedef.h"

G_BEGIN_DECLS
/**
 * @brief Convert to tensors as customized operation
 * @param[in] in_buf the input stream buffer
 * @param[out] config tensors config structure to be filled
 * @return output buffer filled by user
 */
typedef GstBuffer * (* tensor_converter_custom) (GstBuffer *in_buf,
    GstTensorsConfig *config);

/**
 * @brief Register the custom callback function.
 * @param[in] name The name of tensor_converter custom callback function.
 * @param[in] func The custom condition function body
 * @param[in/out] data The internal data for the function
 * @return 0 if success. -ERRNO if error.
 */
extern int
nnstreamer_converter_custom_register (const gchar *name, tensor_converter_custom func, void *data);

/**
 * @brief Unregister the custom callback function.
 * @param[in] name The registered name of tensor_converter custom callback function.
 * @return 0 if success. -ERRNO if error.
 */
extern int
nnstreamer_converter_custom_unregister (const gchar *name);

G_END_DECLS
#endif /*__NNS_TENSOR_CONVERTER_CUSTOM_H__*/
