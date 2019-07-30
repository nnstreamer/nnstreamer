/**
 * NNStreamer API for Tensor_Decoder Sub-Plugins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file  nnstreamer_plugin_api_decoder.h
 * @date  30 Jan 2019
 * @brief Mandatory APIs for NNStreamer Decoder sub-plugins (Need Gst Devel)
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNS_PLUGIN_API_DECODER_H__
#define __NNS_PLUGIN_API_DECODER_H__

#include "tensor_typedef.h"
#include <gst/gst.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Decoder definitions for different semantics of tensors
 *        This allows developers to create their own decoders.
 */
typedef struct _GstTensorDecoderDef
{
  char *modename;
      /**< Unique decoder name. GST users choose decoders with mode="modename". */
  int (*init) (void **private_data);
      /**< Object initialization for the decoder */
  void (*exit) (void **private_data);
      /**< Object destruction for the decoder */
  int (*setOption) (void **private_data, int opNum, const char *param);
      /**< Process with the given options. It can be called repeatedly */
  GstCaps *(*getOutCaps) (void **private_data, const GstTensorsConfig *config);
      /**< The caller should unref the returned GstCaps
        * Current implementation supports single-tensor only.
        * @todo WIP: support multi-tensor for input!!!
        */
  GstFlowReturn (*decode) (void **private_data, const GstTensorsConfig *config,
      const GstTensorMemory *input, GstBuffer *outbuf);
      /**< outbuf must be allocated but empty (gst_buffer_get_size (outbuf) == 0).
        * Note that we support single-tensor (other/tensor) only!
        * @todo WIP: support multi-tensor for input!!!
        */
  size_t (*getTransformSize) (void **private_data, const GstTensorsConfig *config,
      GstCaps *caps, size_t size, GstCaps *othercaps,
      GstPadDirection direction);
      /**< EXPERIMENTAL! @todo We are not ready to use this. This should be NULL or return 0 */
} GstTensorDecoderDef;

/* extern functions for subplugin management, exist in tensor_decoder.c */
/**
 * @brief Decoder's sub-plugin should call this function to register itself.
 * @param[in] decoder Decoder sub-plugin to be registered.
 * @return TRUE if registered. FALSE is failed or duplicated.
 */
extern int
nnstreamer_decoder_probe (GstTensorDecoderDef * decoder);

/**
 * @brief Decoder's sub-plugin may call this to unregister itself.
 * @param[in] name The name of decoder sub-plugin.
 */
extern void
nnstreamer_decoder_exit (const char *name);

/**
 * @brief Find decoder sub-plugin with the name.
 * @param[in] name The name of decoder sub-plugin.
 * @return NULL if not found or the sub-plugin object has an error.
 */
extern const GstTensorDecoderDef *
nnstreamer_decoder_find (const char *name);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_PLUGIN_API_DECODER_H__ */
