/**
 * NNStreamer API for Tensor_Decoder Sub-Plugins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file  nnstreamer_plugin_api_converter.h
 * @date  09 Dec 2019
 * @brief Mandatory APIs for NNStreamer Converter sub-plugins (Need Gst Devel)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_CONVERTER_H__
#define __NNS_PLUGIN_API_CONVERTER_H__

#include "tensor_typedef.h"
#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************
* External Converters                          *
************************************************/

/**
 * @brief Converter's subplugin implementation.
 */
struct _NNStreamerExternalConverter {
  const char *media_type_name;

  /** 1. chain func, data handling. */
  GstBuffer * (*convert) (GstTensorConverter * self,
      const GstBuffer * buf, gsize * frame_size, guint * frames_in);
      /**< Convert the given input stream to tensor/tensors stream.
       * @param[in/out] self A pointer designating "this".
       * @param[in] buf The input stream buffer
       * @param[out] frame_size The size of each frame (output buffer)
       * @param[out] frames_in The number of frames in the given input buffer.
       * @retval Return inbuf if the data is to be kept untouched.
       * @retval Retrun a new GstBuf if the data is to be modified.
       */

  /** 2. parse_caps (type conv, input(media) to output(tensor)) */
  gboolean (*get_caps) (GstTensorConverter * self, const GstStructure * st,
      GstTensorConfig * config);
      /**< Set the tensor config structure from the given stream frame
       * @param[in/out] self A pointer designating "this".
       * @param[in] st The input (original/media data) stream's metadata
       * @param[out] config The output (tensor/tensors) emtadata
       */
  /** 3. query_cap (tpye conf, output(tensor) to input(media)) */
  gboolean (*query_caps) (GstTensorConverter * self,
      const GstTensorConfig *config, GstStructure *st);
      /**< Filters (narrows down) the GstCap (st) with the given config.
       * @param[in/out] self A pointer designating "this".
       * @param[in] config The config of output tensor/tensors
       * @param[in/out] st The gstcap of input to be filtered with config.
       */
};


/**
 * @brief Converter's sub-plugin should call this function to register itself.
 * @param[in] ex Converter sub-plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
extern int
registerExternalConverter (NNStreamerExternalConverter *ex);

/**
 * @brief Converter's sub-plugin may call this to unregister itself.
 * @param[in] prefix The name of converter sub-plugin.
 */
extern void
unregisterExternalConverter (const char *prefix);

#ifdef __cplusplus
}
#endif
#endif /* __NNS_PLUGIN_API_CONVERTER_H__ */
