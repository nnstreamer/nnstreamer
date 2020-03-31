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
 * @file  nnstreamer_plugin_api_decoder.h
 * @date  30 Jan 2019
 * @brief Mandatory APIs for NNStreamer Decoder sub-plugins (Need Gst Devel)
 * @see https://github.com/nnstreamer/nnstreamer
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
      /**< Object initialization for the decoder.
       *
       * @param[in/out] private_data A sub-plugin may save its internal private data here. The sub-plugin is responsible for alloc/free of this pointer. Normally, the sub-plugin may allocate the private_data with this function.
       * @return TRUE if OK. FALSE if error.
       */
  void (*exit) (void **private_data);
      /**< Object destruction for the decoder.
       *
       * @param[in/out] private_data A sub-plugin may save its internal private data here. The sub-plugin is responsible for alloc/free of this pointer. Normally, the sub-plugin may free the private_data with this function.
       */
  int (*setOption) (void **private_data, int opNum, const char *param);
      /**< Process with the given options. It can be called repeatedly.
       *
       * @param[in/out] private_data A sub-plugin may save its internal private data here. The sub-plugin is responsible for alloc/free of this pointer.
       * @param[in] opNum The index of the given options.
       * @param[in] param The option string. A sub-plugin should parse the string to get the proper value.
       * @return TRUE if OK. FALSE if error.
       */
  GstCaps *(*getOutCaps) (void **private_data, const GstTensorsConfig *config);
      /**< The caller should unref the returned GstCaps.
       * The sub-plugin should validate the information of input tensor and return proper media type.
       * Note that the information of input tensor is not a fixed value and the pipeline may try different values during the cap negotiations.
       * Do NOT allocate or fix internal data structure until decode is called.
       *
       * @param[in/out] private_data A sub-plugin may save its internal private data here. The sub-plugin is responsible for alloc/free of this pointer.
       * @param[in] config The structure of input tensor info.
       * @return GstCaps object describing media type.
       */
  GstFlowReturn (*decode) (void **private_data, const GstTensorsConfig *config,
      const GstTensorMemory *input, GstBuffer *outbuf);
      /**< The function to be called when the input tensor incomes into tensor_decoder.
       * The sub-plugin should update the output buffer. outbuf must be allocated but empty (gst_buffer_get_size (outbuf) == 0).
       *
       * @param[in/out] private_data A sub-plugin may save its internal private data here. The sub-plugin is responsible for alloc/free of this pointer.
       * @param[in] config The structure of input tensor info.
       * @param[in] input The array of input tensor data. The maximum array size of input data is NNS_TENSOR_SIZE_LIMIT.
       * @param[out] outbuf A sub-plugin should update or append proper memory for the negotiated media type.
       * @return GST_FLOW_OK if OK.
       */
  size_t (*getTransformSize) (void **private_data, const GstTensorsConfig *config,
      GstCaps *caps, size_t size, GstCaps *othercaps,
      GstPadDirection direction);
      /**< Optional. The sub-plugin may calculate the size in bytes of a buffer.
       * If this is NULL, tensor_decoder will pass the empty buffer and the sub-plugin should append the memory block when called decode.
       * See GstBaseTransformClass::transform_size for the details.
       *
       * @param[in/out] private_data A sub-plugin may save its internal private data here. The sub-plugin is responsible for alloc/free of this pointer.
       * @param[in] config The structure of input tensor info.
       * @param[in] caps GstCaps object for the given direction.
       * @param[in] size The size of a buffer for the given direction.
       * @param[in] othercaps GstCaps object on the other pad for the given direction.
       * @param[in] direction The direction of a pad. Normally this is GST_PAD_SINK.
       * @return The size of a buffer.
       */
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
