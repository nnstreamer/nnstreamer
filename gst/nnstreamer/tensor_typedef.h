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
 * @brief Fixed size of string type
 */
#define GST_TENSOR_STRING_SIZE (1024)
#define NNS_TENSOR_RANK_LIMIT	(4)
#define NNS_TENSOR_SIZE_LIMIT	(16)
#define NNS_TENSOR_SIZE_LIMIT_STR	"16"
#define NNS_TENSOR_DIM_NULL ({0, 0, 0, 0})


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
       but when we call gst_structure_get_string, it actually is working well
     * "dimensions = (string) dim1:dim2:dim3:dim4, dim1:dim2:dim3:dim4"
     */

#define GST_TENSOR_VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, GRAY8 }") \
    ", views = (int) 1, interlace-mode = (string) progressive"

#define GST_TENSOR_AUDIO_CAPS_STR \
    GST_AUDIO_CAPS_MAKE ("{ S8, U8, S16LE, S16BE, U16LE, U16BE, S32LE, S32BE, U32LE, U32BE, F32LE, F32BE, F64LE, F64BE }") \
    ", layout = (string) interleaved"

#define GST_TENSOR_TEXT_CAPS_STR \
    "text/x-raw, format = (string) utf8"

#define GST_TENSOR_OCTET_CAPS_STR \
    "application/octet-stream"

/**
 * @brief Caps string for supported types
 * @todo Support other types
 */
#define GST_TENSOR_MEDIA_CAPS_STR \
    GST_TENSOR_VIDEO_CAPS_STR "; " \
    GST_TENSOR_AUDIO_CAPS_STR "; " \
    GST_TENSOR_TEXT_CAPS_STR "; " \
    GST_TENSOR_OCTET_CAPS_STR

#define GST_TENSOR_TYPE_ALL "{ float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }"

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
 */
typedef enum _nns_media_type
{
  _NNS_VIDEO = 0, /**< supposedly video/x-raw */
  _NNS_AUDIO, /**< supposedly audio/x-raw */
  _NNS_STRING, /**< supposedly text/x-raw */
  _NNS_OCTET, /**< supposedly application/octet-stream */

  _NNS_MEDIA_END, /**< End Marker */
} media_type;

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
 * @brief Internal data structure for configured tensor info (for other/tensor).
 */
typedef struct
{
  GstTensorInfo info; /**< tensor info*/
  int32_t rate_n; /**< framerate is in fraction, which is numerator/denominator */
  int32_t rate_d; /**< framerate is in fraction, which is numerator/denominator */
} GstTensorConfig;

/**
 * @brief Internal data structure for configured tensors info (for other/tensors).
 */
typedef struct
{
  GstTensorsInfo info; /**< tensor info*/
  int32_t rate_n; /**< framerate is in fraction, which is numerator/denominator */
  int32_t rate_d; /**< framerate is in fraction, which is numerator/denominator */
} GstTensorsConfig;


/** @todo Separate headers per subplugin-category */
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

/**
 * @brief Tensor_Filter Subplugin definition
 *
 * Common callback parameters:
 * prop Filter properties. Read Only.
 * private_data Subplugin's private data. Set this (*private_data = XXX) if you want to change filter->private_data.
 */
typedef struct _GstTensorFilterFramework
{
  char *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
  int allow_in_place; /**< TRUE(nonzero) if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
  int allocate_in_invoke; /**< TRUE(nonzero) if invoke_NN is going to allocate outputptr by itself and return the address via outputptr. Do not change this value after cap negotiation is complete (or the stream has been started). */

  int (*invoke_NN) (const GstTensorFilterProperties * prop, void **private_data,
      const GstTensorMemory * input, GstTensorMemory * output);
      /**< Mandatory callback. Invoke the given network model.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] input The array of input tensors. Allocated and filled by tensor_filter/main
       * @param[out] output The array of output tensors. Allocated by tensor_filter/main and to be filled by invoke_NN. If allocate_in_invoke is TRUE, sub-plugin should allocate the memory block for output tensor. (data in GstTensorMemory)
       * @return 0 if OK. non-zero if error.
       */

  int (*getInputDimension) (const GstTensorFilterProperties * prop,
      void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of input tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return the size of input tensors
       */

  int (*getOutputDimension) (const GstTensorFilterProperties * prop,
      void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of output tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return the size of output tensors
       */

  int (*setInputDimension) (const GstTensorFilterProperties * prop,
      void **private_data, const GstTensorsInfo * in_info,
      GstTensorsInfo * out_info);
      /**< Optional. Set Null if not supported. Tensor_Filter::main will
       * configure input dimension from pad-cap in run-time for the sub-plugin.
       * Then, the sub-plugin is required to return corresponding output dimension
       * If this is NULL, both getInput/OutputDimension must be non-NULL.
       *
       * When you use this, do NOT allocate or fix internal data structure based on it
       * until invoke is called. Gstreamer may try different dimensions before
       * settling down.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] in_info structure of input tensor info
       * @param[out] out_info structure of output tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

  int (*open) (const GstTensorFilterProperties * prop, void **private_data);
      /**< Optional. tensor_filter.c will call this before any of other callbacks and will call once before calling close
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, open() allocates memory for private_data.
       * @return 0 if ok. < 0 if error.
       */

  void (*close) (const GstTensorFilterProperties * prop, void **private_data);
      /**< Optional. tensor_filter.c will not call other callbacks after calling close. Free-ing private_data is this function's responsibility. Set NULL after that.
       *
       * @param[in] prop read-only property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, close() frees private_data and set NULL.
       */

  void (*destroyNotify) (void * data);
      /**< Optional. tensor_filter.c will call it when 'allocate_in_invoke' flag of the framework is TRUE. Basically, it is called when the data element is destroyed. If it's set as NULL, g_free() will be used as a default. It will be helpful when the data pointer is included as an object of a nnfw. For instance, if the data pointer is removed when the object is gone, it occurs error. In this case, the objects should be maintained for a while first and destroyed when the data pointer is destroyed. Those kinds of logic could be defined at this method.
       *
       * @param[in] data the data element.
       */
} GstTensorFilterFramework;

#endif /*__GST_TENSOR_TYPEDEF_H__*/
