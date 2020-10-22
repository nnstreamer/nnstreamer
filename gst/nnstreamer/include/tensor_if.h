/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-IF
 * Copyright (C) 2020 Gichan Jang <gichan2.jang@samsung.com>
 */
/**
 * @file	tensor_if.h
 * @date	28 Oct 2020
 * @brief	NNStreamer APIs for tensor_if custom condition
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Gichan Jang <gichan2.jang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensor_if
 *
 * How To for NNdevelopers:
 *
 * Define the function in the app.
 * 1. Define struct, "tensor_if_custom", with the functions defined.
 * 2. Register the struct with "nnstreamer_if_custom_register" API.
 * 3. Construct the nnstreamer pipeline and execute it in the app.
 *
 *
 * Usage example of the tensor_if custom condition
 * @code
 * // Define custom callback function describing the condition
 * gboolean tensor_if_custom_cb (const GstTensorsInfo *info,
 *     const GstTensorMemory * input, gboolean * result) {
 *   // Describe the conditions and pass the results
 * }
 *
 * ...
 * // Register custom callback function describing the condition
 * nnstreamer_if_custom_register ("tifx", tensor_if_custom_cb, NULL);
 * ...
 * // Use the condition in a pipeline.
 * // E.g., Pipeline of " ... ! tensor_if compared-value=CUSTOM compared-value-option=tifx then=TENSORPICK then-option=0 else=TENSORPICK else-option=1 ! ... "
 * ...
 * // After everything is done.
 * nnstreamer_if_custom_unregister ("tifx");
 * @endcode
 */

#ifndef __NNS_TENSOR_IF_H__
#define __NNS_TENSOR_IF_H__

#include <glib.h>
#include <gst/gst.h>
#include "tensor_typedef.h"

G_BEGIN_DECLS
/**
 * @brief Calls the user defined conditions function
 * @param[in] info input tensors info
 * @param[in] input memory containing input tensor data
 * @param[in] user_data private data for the callback
 * @param[out] result result of the user defined condition
 * @return TRUE if there is no error.
 */
typedef gboolean (* tensor_if_custom) (const GstTensorsInfo *info,
    const GstTensorMemory *input, void *user_data, gboolean *result);

/**
 * @brief Register the custom callback function.
 * @param[in] name The name of tensor_if custom callback function.
 * @param[in] func The custom condition function body
 * @param[in/out] data The internal data for the function
 * @return 0 if success. -ERRNO if error.
 * @note GstElement`tensor_if_h` can be get with `gst_bin_get_by_name` from the pipeline.
 *       e.g.,) pipeline description: ... ! tensor_if name=tif (... properties ...) ! ...
 *       GstElement *tensor_if_h = gst_bin_get_by_name (GST_BIN (pipeline), "tif");
 */
extern int
nnstreamer_if_custom_register (const gchar *name, tensor_if_custom func, void *data);

/**
 * @brief Unregister the custom callback function.
 * @param[in] name The registered name of tensor_if custom callback function.
 * @return 0 if success. -ERRNO if error.
 */
extern int
nnstreamer_if_custom_unregister (const gchar *name);

G_END_DECLS
#endif /*__NNS_TENSOR_IF_H__*/
