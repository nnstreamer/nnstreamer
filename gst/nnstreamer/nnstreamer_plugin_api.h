/**
 * NNStreamer Common API Header for Plug-Ins
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
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
 * @file  nnstreamer_plugin_api.h
 * @date  24 Jan 2019
 * @brief Header file that contains typedefed data types and APIs for NNStreamer plug-ins
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com> and Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */
#ifndef __NNS_PLUGIN_API_H__
#define __NNS_PLUGIN_API_H__

#include <glib.h>
#include <gst/gst.h>
#include <gst/video/video-format.h>
#include <gst/audio/audio-format.h>
#include <tensor_typedef.h>

G_BEGIN_DECLS

/**
 * @brief Fixed size of string type
 */
#ifndef GST_TENSOR_STRING_SIZE
#define GST_TENSOR_STRING_SIZE (1024)
#endif

#define GST_TENSOR_RATE_RANGE "(fraction) [ 0, max ]"

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

/**
 * @brief This value, 16, can be checked with gst_buffer_get_max_memory(),
 * which is GST_BUFFER_MEM_MAX in gstreamer/gstbuffer.c.
 * We redefined the value because GST_BUFFER_MEM_MAX is not exported and
 * we need static value. To modify (increase) this value, you need to update
 * gstreamer/gstbuffer.c as well.
 */
#define GST_TENSOR_NUM_TENSORS_RANGE "(int) [ 1, " NNS_TENSOR_SIZE_LIMIT_STR " ]"

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
 * @brief Internal data structure for configured tensor info (for other/tensor).
 */
typedef struct
{
  GstTensorInfo info; /**< tensor info*/
  gint rate_n; /**< framerate is in fraction, which is numerator/denominator */
  gint rate_d; /**< framerate is in fraction, which is numerator/denominator */
} GstTensorConfig;

/**
 * @brief Internal data structure for configured tensors info (for other/tensors).
 */
typedef struct
{
  GstTensorsInfo info; /**< tensor info*/
  gint rate_n; /**< framerate is in fraction, which is numerator/denominator */
  gint rate_d; /**< framerate is in fraction, which is numerator/denominator */
} GstTensorsConfig;

/**
 * @brief String representations for each tensor element type.
 */
extern const gchar *tensor_element_typename[];

/**
 * @brief Initialize the tensor info structure
 * @param info tensor info structure to be initialized
 */
extern void
gst_tensor_info_init (GstTensorInfo * info);

/**
 * @brief Get data size of single tensor
 * @param info tensor info structure
 * @return data size
 */
extern gsize
gst_tensor_info_get_size (const GstTensorInfo *info);

/**
 * @brief Check the tensor info is valid
 * @param info tensor info structure
 * @return TRUE if info is valid
 */
extern gboolean
gst_tensor_info_validate (const GstTensorInfo *info);

/**
 * @brief Initialize the tensors info structure
 * @param info tensors info structure to be initialized
 */
extern void
gst_tensors_info_init (GstTensorsInfo *info);

/**
 * @brief Parse the string of dimensions
 * @param info tensors info structure
 * @param dim_string string of dimensions
 * @return number of parsed dimensions
 */
extern guint
gst_tensors_info_parse_dimensions_string (GstTensorsInfo *info,
    const gchar *dim_string);

/**
 * @brief Parse the string of types
 * @param info tensors info structure
 * @param type_string string of types
 * @return number of parsed types
 */
extern guint
gst_tensors_info_parse_types_string (GstTensorsInfo * info,
    const gchar * type_string);

/**
 * @brief Check the tensors info is valid
 * @param info tensors info structure
 * @return TRUE if info is valid
 */
extern gboolean
gst_tensors_info_validate (const GstTensorsInfo *info);

/**
 * @brief Initialize the tensor config info structure
 * @param config tensor config structure to be initialized
 */
extern void
gst_tensor_config_init (GstTensorConfig * config);

/**
 * @brief Check the tensor is all configured
 * @param config tensor config structure
 * @return TRUE if configured
 */
extern gboolean
gst_tensor_config_validate (const GstTensorConfig * config);

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
extern media_type
gst_tensor_media_type_from_structure (const GstStructure * structure);

/**
 * @brief Parse structure and set tensor config info (for other/tensor)
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_config_from_structure (GstTensorConfig *config,
    const GstStructure *structure);

/**
 * @brief Parse structure and set tensors config (for other/tensors)
 * @param config tensors config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if no error
 */
extern gboolean
gst_tensors_config_from_structure (GstTensorsConfig *config,
    const GstStructure *structure);

/**
 * @brief Initialize the tensors config info structure (for other/tensors)
 * @param config tensors config structure to be initialized
 */
extern void
gst_tensors_config_init (GstTensorsConfig * config);

/**
 * @brief Check the tensors are all configured (for other/tensors)
 * @param config tensor config structure
 * @return TRUE if configured
 */
extern gboolean
gst_tensors_config_validate (const GstTensorsConfig * config);

/**
 * @brief Check the tensor dimension is valid
 * @param dim tensor dimension
 * @return TRUE if dimension is valid
 */
extern gboolean
gst_tensor_dimension_is_valid (const tensor_dim dim);

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param dimstr The dimension string in the format of d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d4][d3][d2][d1];
 * @param dim dimension to be filled.
 */
extern int get_tensor_dimension (const gchar * dimstr, tensor_dim dim);

/**
 * @brief Get dimension string from given tensor dimension.
 * @param dim tensor dimension
 * @return Formatted string of given dimension (d1:d2:d3:d4).
 * @note The returned value should be freed with g_free()
 */
extern gchar *get_tensor_dimension_string (const tensor_dim dim);

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
extern size_t get_tensor_element_count (const tensor_dim dim);

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
extern tensor_type get_tensor_type (const gchar * typestr);

G_END_DECLS
#endif /* __NNS_PLUGIN_API_H__ */
