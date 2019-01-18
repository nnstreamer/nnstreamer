/**
 * NNStreamer Common Header, Typedef part, for export as devel package.
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_typedef.h
 * @date	01 Jun 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * To Packagers:
 *
 * This fils it to be packaged as "devel" package for NN developers.
 */
#ifndef __GST_TENSOR_TYPEDEF_H__
#define __GST_TENSOR_TYPEDEF_H__

#include <stddef.h>
#include <stdint.h>

#define NNS_TENSOR_RANK_LIMIT	(4)
#define NNS_TENSOR_SIZE_LIMIT	(16)
#define NNS_TENSOR_SIZE_LIMIT_STR	"16"
#define NNS_TENSOR_DIM_NULL ({0, 0, 0, 0})

/**
 * @brief Possible data element types of other/tensor.
 */
typedef enum _nns_tensor_type
{
  _NNS_INT32 = 0,
  _NNS_UINT32,
  _NNS_INT16,
  _NNS_UINT16,
  _NNS_INT8,
  _NNS_UINT8,
  _NNS_FLOAT64,
  _NNS_FLOAT32,
  _NNS_INT64,
  _NNS_UINT64,

  _NNS_END,
} tensor_type;

/**
 * @brief Byte-per-element of each tensor element type.
 */
static const unsigned int tensor_element_size[] = {
  [_NNS_INT32] = 4,
  [_NNS_UINT32] = 4,
  [_NNS_INT16] = 2,
  [_NNS_UINT16] = 2,
  [_NNS_INT8] = 1,
  [_NNS_UINT8] = 1,
  [_NNS_FLOAT64] = 8,
  [_NNS_FLOAT32] = 4,
  [_NNS_INT64] = 8,
  [_NNS_UINT64] = 8,

  [_NNS_END] = 0,
};

/**
 * @brief To make the code simple with all the types. "C++ Template"-like.
 */
typedef union {
  int32_t _int32_t;
  uint32_t _uint32_t;
  int16_t _int16_t;
  uint16_t _uint16_t;
  int8_t _int8_t;
  uint8_t _uint8_t;
  double _double;
  float _float;
  int64_t _int64_t;
  uint64_t _uint64_t;
} tensor_element;

typedef uint32_t tensor_dim[NNS_TENSOR_RANK_LIMIT];

/**
 * @brief The unit of each data tensors. It will be used as an input/output tensor of other/tensors.
 */
typedef struct
{
  void *data;
  size_t size;
  tensor_type type;
} GstTensorMemory;

/**
 * @brief Internal data structure for tensor info.
 */
typedef struct
{
  char * name; /**< Name of each element in the tensor. User must designate this. */
  tensor_type type; /**< Type of each element in the tensor. User must designate this. */
  tensor_dim dimension; /**< Dimension. We support up to 4th ranks.  */
} GstTensorInfo;

/**
 * @brief Internal meta data exchange format for a other/tensors instance
 */
typedef struct
{
  unsigned int num_tensors; /**< The number of tensors */
  GstTensorInfo info[NNS_TENSOR_SIZE_LIMIT]; /**< The list of tensor info */
} GstTensorsInfo;

/**
 * @brief GstTensorFilter's properties for NN framework (internal data structure)
 *
 * Because custom filters of tensor_filter may need to access internal data
 * of GstTensorFilter, we define this data structure here.
 */
typedef struct _GstTensorFilterProperties
{
  const char *fwname; /**< The name of NN Framework */
  int fw_opened; /**< TRUE IF open() is called or tried. Use int instead of gboolean because this is refered by custom plugins. */
  const char *model_file; /**< Filepath to the model file (as an argument for NNFW). char instead of gchar for non-glib custom plugins */

  int input_configured; /**< TRUE if input tensor is configured. Use int instead of gboolean because this is refered by custom plugins. */
  GstTensorsInfo input_meta; /**< configured input tensor info */

  int output_configured; /**< TRUE if output tensor is configured. Use int instead of gboolean because this is refered by custom plugins. */
  GstTensorsInfo output_meta; /**< configured output tensor info */

  const char *custom_properties; /**< sub-plugin specific custom property values in string */
} GstTensorFilterProperties;

#endif /*__GST_TENSOR_TYPEDEF_H__*/
