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

#ifdef __cplusplus
}
#endif

#endif /* _TENSOR_CONVERTER_UTIL_H_ */
