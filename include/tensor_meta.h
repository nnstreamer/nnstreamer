/*
 * GStreamer Tensor Meta
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	tensor_meta.h
 * @date	20 June 2018
 * @brief	Meta Data for Tensor type.
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __GST_TENSOR_META_H__
#define __GST_TENSOR_META_H__

#include <glib.h>
#include <stdint.h>
#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS

typedef struct _GstMetaTensor GstMetaTensor;

/**
 * @brief Definition of Tensor Meta Data
 */
struct _GstMetaTensor {
  GstMeta meta;
  gint num_tensors;
  GList *dimensions;
};

/**
 * @brief Get tensor meta data type. Register Tensor Meta Data API definition
 * @return Tensor Meta Data Type
 */
GType gst_meta_tensor_api_get_type (void);

#define GST_META_TENSOR_API_TYPE (gst_meta_tensor_api_get_type ())

/**
 * @brief get tensor meta data info
 * @return Tensor Meta Data Info
 */
const GstMetaInfo *gst_meta_tensor_get_info (void);
#define GST_META_TENSOR_INFO ((GstMetaInfo*) gst_meta_tensor_get_info ())

/**
 * @brief Macro to get tensor meta data.
 */
#define gst_buffer_get_meta_tensor(b) \
  ((GstMetaTensor*) gst_buffer_get_meta ((b), GST_META_TENSOR_API_TYPE))

/**
 * @brief Add tensor meta data
 * @param buffer The buffer to save meta data
 * @param variable to save meta ( number of tensors )
 * @return Tensor Meta Data
 */
GstMetaTensor * gst_buffer_add_meta_tensor (GstBuffer *buffer);

#define GST_META_TENSOR_GET(buf) ((GstMetaTensor *)gst_buffer_get_meta_tensor (buf))
#define GST_META_TENSOR_ADD(buf) ((GstMetaTensor *)gst_buffer_add_meta_tensor (buf))

/**
 * @brief Utility function to make tensors by add Gst Tensor Meta & Initialize Meta
 * @param buffer Target GstBuffer Object
 * @return GstMetaTensor
 */
GstMetaTensor * gst_make_tensors (GstBuffer *buffer);

/**
 * @brief Utility function to add tensor in tensors.
 *        Add GstMemory for tensor, increase num_tensors and add tensor_dim
 * @param buffer Target GstBuffer Object
 * @param mem GstMemory Object for tensor
 * @param dim tensor_dim for tensor
 * @return GstMetaTensor
 */
GstMetaTensor * gst_append_tensor (GstBuffer *buffer, GstMemory *mem, tensor_dim *dim);

/**
 * @brief Utility function to get tensor from tensors.
 * @param buffer Target GstBuffer Object
 * @param nth order of tensor
 * @return GstMemroy Tensor data
 */
GstMemory * gst_get_tensor (GstBuffer *buffer, gint nth);

/**
 * @brief Utility function to get nth tensor dimension
 * @param buffer Target GstBuffer Object
 * @param nth order of tensor
 * @return tensor_dim Tensor dimension
 */
tensor_dim * gst_get_tensordim (GstBuffer *buffer, gint nth);

/**
 * @brief Utility function to remove nth tensor from tensors
 * @param buffer Target GstBuffer Object
 * @param nth order of tensor
 * @return GstFlowReturn TRUE/FALSE
 */
GstFlowReturn gst_remove_tensor (GstBuffer *buffer, gint nth);

/**
 * @brief Utility function to get number of tensor in tensors
 * @param buffer Target GstBuffer Object
 * @return gint number of tensor
 */
gint gst_get_num_tensors (GstBuffer *buffer);

/**
 * @brief Utility function to get parse the dimension of tensors
 * @param dim_string Input String to parse
 * @return GArray Array which includes tensor_dim for each tensor
 */
GArray * parse_dimensions (const gchar* dim_string);

/**
 * @brief Utility function to get parse the type of tensors
 * @param type_string Input String to parse
 * @return GArray Array which includes tensor_type for each tensor
 */
GArray * parse_types (const gchar* type_string);

G_END_DECLS

#endif /* __GST_TENSOR_META_H__ */
