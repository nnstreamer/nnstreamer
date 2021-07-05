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
 * @brief The maximum rank in meta info (see GstTensorMetaInfo).
 * This RANK is applied to meta info of other/tensors-flexible only and
 * does not affect other/tensors(s)'s NNS_TENSOR_RANK_LIMIT.
 */
#define NNS_TENSOR_META_RANK_LIMIT	(16)

#define NNS_MIMETYPE_TENSOR "other/tensor"
#define NNS_MIMETYPE_TENSORS "other/tensors"

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
 * @brief Possible tensor formats
 */
#define GST_TENSOR_FORMAT_ALL "{ static, flexible, sparse }"

/**
 * @brief Default static capability for other/tensor
 */
#define GST_TENSOR_CAP_DEFAULT \
    NNS_MIMETYPE_TENSOR ", " \
    "framerate = " GST_TENSOR_RATE_RANGE
    /**
     * type should be one of types in GST_TENSOR_TYPE_ALL
     * "type = (string) uint8"
     * dimension should be a formatted string with rank NNS_TENSOR_RANK_LIMIT
     * "dimension = (string) dim1:dim2:dim3:dim4"
     */

/**
 * @brief Caps string for the caps template of tensor stream.
 * format should be a string that describes the data format, or possible formats of incoming tensor.
 *
 * If the data format is static, another tensor information should be described in caps.
 * - num_tensors: The number of tensors in a GstBuffer.
 * - types: The list of data type in the tensor. type should be one of types in GST_TENSOR_TYPE_ALL. (types=(string)"type1,type2,type3")
 * - dimensions: The list of dimension in the tensor. dimension should be a formatted string with rank NNS_TENSOR_RANK_LIMIT. (dimensions=(string)"dim1:dim2:dim3:dim4,dim1:dim2:dim3:dim4")
 */
#define GST_TENSORS_CAP_MAKE(fmt) \
    NNS_MIMETYPE_TENSORS ", " \
    "format = (string) " fmt ", " \
    "framerate = " GST_TENSOR_RATE_RANGE

/**
 * @brief Caps string for the caps template (other/tensors, static tensor stream with fixed number of tensors).
 * num should be a string format that describes the number of tensors, or the range of incoming tensors.
 * The types and dimensions of tensors should be described for caps negotiation.
 */
#define GST_TENSORS_CAP_WITH_NUM(num) \
    NNS_MIMETYPE_TENSORS ", " \
    "format = (string) static, num_tensors = " num ", " \
    "framerate = " GST_TENSOR_RATE_RANGE

/**
 * @brief Caps string for the caps template of static tensor stream.
 */
#define GST_TENSORS_CAP_DEFAULT \
    GST_TENSORS_CAP_WITH_NUM(GST_TENSOR_NUM_TENSORS_RANGE)

/**
 * @brief Caps string for the caps template of flexible tensors.
 * This mimetype handles non-static, flexible tensor stream without specifying the data type and shape of the tensor.
 * The maximum number of tensors in a buffer is 16 (NNS_TENSOR_SIZE_LIMIT).
 */
#define GST_TENSORS_FLEX_CAP_DEFAULT \
    GST_TENSORS_CAP_MAKE ("flexible")

/**
 * @brief Default static capability for Protocol Buffers
 * protobuf converter will convert this capability to other/tensor(s)
 */
#define GST_PROTOBUF_TENSOR_CAP_DEFAULT \
    "other/protobuf-tensor, " \
    "framerate = " GST_TENSOR_RATE_RANGE

/**
 * @brief Default static capability for flatbuffers
 * Flatbuf converter will convert this capability to other/tensor(s)
 * @todo Move this definition to proper header file
 */
#define GST_FLATBUF_TENSOR_CAP_DEFAULT \
    "other/flatbuf-tensor, " \
    "framerate = " GST_TENSOR_RATE_RANGE

/**
 * @brief Default static capability for flexbuffers
 */
#define GST_FLEXBUF_CAP_DEFAULT "other/flexbuf"

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
 * This is related with media input stream to other/tensor.
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
  _NNS_TENSOR = 4, /**< supposedly other/tensor(s) or flexible tensor */
  _NNS_MEDIA_ANY = 0x1000, /**< any media type (find proper external converter in tensor-converter element) */
} media_type;

/**
 * @brief Data format of tensor stream in the pipeline.
 */
typedef enum _tensor_format
{
  _NNS_TENSOR_FORMAT_STATIC = 0,
  _NNS_TENSOR_FORMAT_FLEXIBLE,
  _NNS_TENSOR_FORMAT_SPARSE,

  _NNS_TENSOR_FORMAT_END
} tensor_format;

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
 * @note This must be coherent with api/capi/include/nnstreamer-capi-private.h:ml_tensor_data_s
 */
typedef struct
{
  void *data; /**< The instance of tensor data. */
  size_t size; /**< The size of tensor. */
} GstTensorMemory;

/**
 * @brief Internal data structure for tensor info.
 * @note This must be coherent with api/capi/include/nnstreamer-capi-private.h:ml_tensor_info_s
 */
typedef struct
{
  char *name; /**< Name of each element in the tensor.
                   User must designate this in a few NNFW frameworks (tensorflow)
                   and some (tensorflow-lite) do not need this. */
  tensor_type type; /**< Type of each element in the tensor. User must designate this. */
  tensor_dim dimension; /**< Dimension. We support up to 4th ranks.  */
  tensor_format format; /**< Tensor format */
} GstTensorInfo;

/**
 * @brief Internal meta data exchange format for a other/tensors instance
 * @note This must be coherent with api/capi/include/nnstreamer-capi-private.h:ml_tensors_info_s
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

/**
 * @brief Data structure to describe a tensor data.
 * This represents the basic information of a memory block for tensor stream.
 *
 * Internally NNStreamer handles a buffer with capability other/tensors-flexible using this information.
 * - version: The version of tensor meta.
 * - type: The type of each element in the tensor. This should be a value of enumeration tensor_type.
 * - dimension: The dimension of tensor. This also denotes the rank of tensor. (e.g., [3:224:224:0] means rank 3.)
 * - format: The data format in the tensor. This should be a value of enumeration tensor_format.
 * - media_type: The media type of tensor. This should be a value of enumeration media_type.
 */
typedef struct
{
  uint32_t version;
  uint32_t type;
  uint32_t dimension[NNS_TENSOR_META_RANK_LIMIT];
  uint32_t format;
  uint32_t media_type;
} GstTensorMetaInfo;

#endif /*__GST_TENSOR_TYPEDEF_H__*/
