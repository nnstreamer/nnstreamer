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
 * This file is to be packaged as "devel" package for NN developers.
 */
#ifndef __GST_TENSOR_TYPEDEF_H__
#define __GST_TENSOR_TYPEDEF_H__

#include <stddef.h>
#include <stdint.h>

#define NNS_TENSOR_RANK_LIMIT	(16)

/**
 * @brief The number of tensors NNStreamer supports is 256.
 * The max memories of gst-buffer is 16 (See NNS_TENSOR_MEMORY_MAX).
 * Internally NNStreamer handles the memories as tensors.
 * If the number of tensors is larger than 16, we modify the last memory and combine tensors into the memory.
 */
#define NNS_TENSOR_SIZE_LIMIT		(256)
#define NNS_TENSOR_SIZE_LIMIT_STR	"256"

/**
 * @brief This value, 16, can be checked with gst_buffer_get_max_memory(),
 * which is GST_BUFFER_MEM_MAX in gstreamer/gstbuffer.c.
 * We redefined the value because GST_BUFFER_MEM_MAX is not exported and
 * we need static value. To modify (increase) this value, you need to update
 * gstreamer/gstbuffer.c as well.
 */
#define NNS_TENSOR_MEMORY_MAX		(16)

/**
 * @brief Max number of extra tensors.
 */
#define NNS_TENSOR_SIZE_EXTRA_LIMIT (NNS_TENSOR_SIZE_LIMIT - NNS_TENSOR_MEMORY_MAX)

#define NNS_MIMETYPE_TENSOR "other/tensor"
#define NNS_MIMETYPE_TENSORS "other/tensors"

#define GST_TENSOR_NUM_TENSORS_RANGE "(int) [ 1, " NNS_TENSOR_SIZE_LIMIT_STR " ]"
#define GST_TENSOR_RATE_RANGE "(fraction) [ 0, max ]"

/**
 * @brief Possible tensor element types
 */
#define GST_TENSOR_TYPE_ALL "{ float16, float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }"

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
    GST_TENSORS_CAP_WITH_NUM (GST_TENSOR_NUM_TENSORS_RANGE)

/**
 * @brief Caps string for the caps template of flexible tensors.
 * This mimetype handles non-static, flexible tensor stream without specifying the data type and shape of the tensor.
 * The maximum number of tensors in a buffer is 16 (NNS_TENSOR_SIZE_LIMIT).
 */
#define GST_TENSORS_FLEX_CAP_DEFAULT \
    GST_TENSORS_CAP_MAKE ("flexible")

/**
 * @brief Caps string for the caps template of sparse tensors.
 * This mimetype handles non-static, sparse tensor stream without specifying the data type and shape of the tensor.
 * The maximum number of tensors in a buffer is 16 (NNS_TENSOR_SIZE_LIMIT).
 */
#define GST_TENSORS_SPARSE_CAP_DEFAULT \
    GST_TENSORS_CAP_MAKE ("sparse")

/**
 * @brief Possible data element types of other/tensor.
 * @note When changing tensor type, you should update related type in ML-API and protobuf/flatbuf schema also.
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
  _NNS_FLOAT16, /**< added with nnstreamer 2.1.1-devel. If you add any operators (e.g., tensor_transform) to float16, it will either be not supported or be too inefficient. */

  _NNS_END,
} tensor_type;

/**
 * @brief Float16 compiler extension support
 */
#if defined(FLOAT16_SUPPORT)
#if defined(__aarch64__) || defined(__arm__)
/** arm (32b) requires "-mfp16-format=ieee */
typedef __fp16 float16;
#elif defined(__x86_64) || defined(__i686__)
/** recommends "-mavx512fp16" for hardware acceleration (gcc>=12 x64)*/
typedef _Float16 float16;
#else
#error "Float 16 supported only with aarch64, arm, x86/64. In arm, you need -mfp16-format=ieee"
#endif /* x86/64, arm64/arm32 */
#endif /* FLOAT16_SUPPORT */

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
 * @brief Tensor layout format for other/tensor
 *
 * The layout is needed by some of the element to appropriately process the
 * data based on the axis of the channel in the data. Layout information will be
 * currently utilized by only some of the elements (SNAP, NNFW in tensor_filter,
 * PADDING mode in tensor_transform)
 *
 * Tensor layout is not part of the capabilities of the element,
 * and does not take part in the caps negotiation.
 *
 * NONE layout implies that the layout of the data is neither NHWC nor NCHW. '
 * However, ANY layout implies that the layout of the provided data is not
 * relevant.
 *
 * @note Providing tensor layout can also decide acceleration to be supported
 * as not all the accelerators might support all the layouts (NYI).
 */
typedef enum _nns_tensor_layout
{
  _NNS_LAYOUT_ANY = 0,     /**< does not care about the data layout */
  _NNS_LAYOUT_NHWC,        /**< NHWC: channel last layout */
  _NNS_LAYOUT_NCHW,        /**< NCHW: channel first layout */
  _NNS_LAYOUT_NONE,        /**< NONE: none of the above defined layouts */
} tensor_layout;

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
#ifdef FLOAT16_SUPPORT
  float16 _float16;
#endif
} tensor_element;

typedef uint32_t tensor_dim[NNS_TENSOR_RANK_LIMIT];

/**
 * @brief The unit of each data tensors. It will be used as an input/output tensor of other/tensors.
 */
typedef struct
{
  void *data; /**< The instance of tensor data. */
  size_t size; /**< The size of tensor. */
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
  tensor_dim dimension; /**< Dimension. We support up to 16th ranks.  */
} GstTensorInfo;

/**
 * @brief Internal meta data exchange format for a other/tensors instance
 */
typedef struct
{
  unsigned int num_tensors; /**< The number of tensors */
  GstTensorInfo info[NNS_TENSOR_MEMORY_MAX]; /**< The list of tensor info (max NNS_TENSOR_MEMORY_MAX as static) */
  GstTensorInfo *extra; /**< The list of tensor info for tensors whose index is larger than NNS_TENSOR_MEMORY_MAX */
  tensor_format format; /**< tensor stream type */
} GstTensorsInfo;

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
 * @brief Internal data structure for sparse tensor info
 */
typedef struct
{
  uint32_t nnz; /**< the number of "non-zero" elements */
} GstSparseTensorInfo;

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
  uint32_t magic;
  uint32_t version;
  uint32_t type;
  tensor_dim dimension;
  uint32_t format;
  uint32_t media_type;

  /**
   * @brief Union of the required information for processing each tensor "format".
   */
  union {
    GstSparseTensorInfo sparse_info;
  };

} GstTensorMetaInfo;

#endif /*__GST_TENSOR_TYPEDEF_H__*/
