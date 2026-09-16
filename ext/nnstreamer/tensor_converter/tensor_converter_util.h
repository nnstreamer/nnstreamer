/* SPDX-License-Identifier: LGPL-2.1-only */
/**
* @file        tensor_converter_util.h
* @date        26 May 2021
* @brief       Utility functions for NNStreamer tensor-converter subplugins.
* @see         https://github.com/nnstreamer/nnstreamer
* @author      MyungJoo Ham <myungjoo.hamt@samsung.com>
* @bug         No known bugs except for NYI items
*/
#ifndef _TENSOR_CONVERTER_UTIL_H_
#define _TENSOR_CONVERTER_UTIL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>
#include <nnstreamer_plugin_api.h>

/** @brief tensor converter plugin's NNStreamerExternalConverter callback */
gboolean tcu_get_out_config (const GstCaps *in_cap, GstTensorsConfig *config);

/**
 * @brief Check that the data a serialized stream carries for a tensor holds what the stream declares for it.
 * @param config The tensors config parsed from the stream. Its format decides whether @a data starts with a meta header.
 * @param info The tensor info parsed from the stream.
 * @param data The data of the tensor in the stream.
 * @param size The size of @a data in bytes.
 * @return TRUE if @a data may be handed downstream as the tensor described by @a info.
 * @note ext/nnstreamer/extra/nnstreamer_protobuf.cc keeps a copy of this check; change both together.
 */
gboolean tcu_check_tensor_data (const GstTensorsConfig *config,
    const GstTensorInfo *info, const guint8 *data, gsize size);

#ifdef __cplusplus
}
#endif

#endif /* _TENSOR_CONVERTER_UTIL_H_ */
