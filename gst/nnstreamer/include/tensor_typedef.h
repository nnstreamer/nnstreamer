/**
 * NNStreamer Common Header, Typedef part, for export as devel package.
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_typedef.h
 * @date	01 Jun 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/nnstreamer/nnstreamer
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
 * @brief This value, 16, can be checked with gst_buffer_get_max_memory(),
 * which is GST_BUFFER_MEM_MAX in gstreamer/gstbuffer.c.
 * We redefined the value because GST_BUFFER_MEM_MAX is not exported and
 * we need static value. To modify (increase) this value, you need to update
 * gstreamer/gstbuffer.c as well.
 */
#define GST_TENSOR_NUM_TENSORS_RANGE "(int) [ 1, " NNS_TENSOR_SIZE_LIMIT_STR " ]"
#define GST_TENSOR_RATE_RANGE "(fraction) [ 0, max ]"

/**
 * @brief Possible tensor element types
 */
#define GST_TENSOR_TYPE_ALL "{ float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }"

/**
 * @brief Default static capibility for other/tensor
 */
#define GST_TENSOR_CAP_DEFAULT \
    "other/tensor, " \
    "framerate = " GST_TENSOR_RATE_RANGE
    /**
     * type should be one of types in GST_TENSOR_TYPE_ALL
     * "type = (string) uint8"
     * dimension shoule be a formatted string with rank NNS_TENSOR_RANK_LIMIT
     * "dimension = (string) dim1:dim2:dim3:dim4"
     */

/**
 * @brief Default static capibility for other/tensors
 */
#define GST_TENSORS_CAP_DEFAULT \
    "other/tensors, " \
    "num_tensors = " GST_TENSOR_NUM_TENSORS_RANGE ", "\
    "framerate = " GST_TENSOR_RATE_RANGE
    /**
     * type should be one of types in GST_TENSOR_TYPE_ALL
     * "types = (string) uint8, uint8, uint8"
     * Dimensions of Tensors for negotiation. It's comment out here,
     * but when we call gst_structure_get_string, it actually is working well.
     * "dimensions = (string) dim1:dim2:dim3:dim4, dim1:dim2:dim3:dim4"
     */

/**
 * @brief Default static capibility for flatbuffers
 * Flatbuf converter will convert this capability to other/tensor(s)
 * @todo Move this definition to proper header file
 */
#define GST_FLATBUF_TENSOR_CAP_DEFAULT \
    "other/flatbuf-tensor, " \
    "framerate = " GST_TENSOR_RATE_RANGE

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
 * @brief Possible input stream types for other/tensor.
 *
 * This is realted with media input stream to other/tensor.
 * There is no restrictions for the outputs.
 *
 * In order to prevent enum-mix issues between device profiles,
 * we explicitly define numbers for each enum type.
 */
typedef enum _nns_media_type
{
  _NNS_MEDIA_INVALID = -1, /**< Uninitialized */
  _NNS_VIDEO = 0, /**< supposedly video/x-raw */
  _NNS_AUDIO = 1, /**< supposedly audio/x-raw */
  _NNS_TEXT = 2, /**< supposedly text/x-raw */
  _NNS_OCTET = 3, /**< supposedly application/octet-stream */
  _NNS_MEDIA_PLUGINS = 0x1000, /**< external converters */
} media_type;

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
  char *name; /**< Name of each element in the tensor.
                   User must designate this in a few NNFW frameworks (tensorflow)
                   and some (tensorflow-lite) do not need this. */
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
 * @brief Internal data structure for configured tensor info (for other/tensor).
 */
typedef struct
{
  GstTensorInfo info; /**< tensor info*/
  int rate_n; /**< framerate is in fraction, which is numerator/denominator */
  int rate_d; /**< framerate is in fraction, which is numerator/denominator */
} GstTensorConfig;

/**
 * @brief Internal data structure for configured tensors info (for other/tensors).
 */
typedef struct
{
  GstTensorsInfo info; /**< tensor info*/
  int rate_n; /**< framerate is in fraction, which is numerator/denominator */
  int rate_d; /**< framerate is in fraction, which is numerator/denominator */
} GstTensorsConfig;

#endif /*__GST_TENSOR_TYPEDEF_H__*/
