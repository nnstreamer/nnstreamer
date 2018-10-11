/**
 * NNStreamer Common Header
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
 * @file	tensor_common.h
 * @date	23 May 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __GST_TENSOR_COMMON_H__
#define __GST_TENSOR_COMMON_H__

#include <glib.h>
#include <stdint.h>
#include "tensor_typedef.h"
#include <gst/gst.h>
#include <gst/video/video-format.h>
#include <gst/audio/audio-format.h>
#include <gst/gstplugin.h>

G_BEGIN_DECLS

/**
 * @brief Fixed size of string type
 */
#ifndef GST_TENSOR_STRING_SIZE
#define GST_TENSOR_STRING_SIZE (1024)
#endif

#define GST_TENSOR_VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ RGB, BGRx, GRAY8 }") \
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

/** @todo I'm not sure if the range is to be 1, 65535 or larger */
#define GST_TENSOR_DIM_RANGE "(int) [ 1, 65535 ]"
#define GST_TENSOR_RATE_RANGE "(fraction) [ 0/1, 2147483647/1 ]"
#define GST_TENSOR_TYPE_ALL "{ float32, float64, int64, uint64, int32, uint32, int16, uint16, int8, uint8 }"

#define GST_TENSOR_CAP_DEFAULT \
    "other/tensor, " \
    "dim1 = " GST_TENSOR_DIM_RANGE ", " \
    "dim2 = " GST_TENSOR_DIM_RANGE ", " \
    "dim3 = " GST_TENSOR_DIM_RANGE ", " \
    "dim4 = " GST_TENSOR_DIM_RANGE ", " \
    "type = (string) " GST_TENSOR_TYPE_ALL ", " \
    "framerate = " GST_TENSOR_RATE_RANGE


/**
 * @brief This value, 16, can be checked with gst_buffer_get_max_memory(),
 * which is GST_BUFFER_MEM_MAX in gstreamer/gstbuffer.c.
 * We redefined the value because GST_BUFFER_MEM_MAX is not exported and
 * we need static value. To modify (increase) this value, you need to update
 * gstreamer/gstbuffer.c as well.
 */
#define GST_TENSOR_NUM_TENSORS_RANGE "(int) [ 1, " NNS_TENSOR_SIZE_LIMIT_STR " ]"

/**
 * @brief Default static capibility for other/tensors
 *
 * This type uses GstMetaTensor to describe tensor. So there is no need to ask information
 * to identify each tensor.
 *
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
 * @brief Get media type from caps
 * @param caps caps to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
extern media_type
gst_tensor_media_type_from_caps (const GstCaps * caps);

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
extern media_type
gst_tensor_media_type_from_structure (const GstStructure * structure);

/**
 * @brief Initialize the tensor info structure
 * @param info tensor info structure to be initialized
 */
extern void
gst_tensor_info_init (GstTensorInfo * info);

/**
 * @brief Check the tensor info is valid
 * @param info tensor info structure
 * @return TRUE if info is valid
 */
extern gboolean
gst_tensor_info_validate (const GstTensorInfo * info);

/**
 * @brief Compare tensor info
 * @param TRUE if equal
 */
extern gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2);

/**
 * @brief Get data size of single tensor
 * @param info tensor info structure
 * @return data size
 */
extern gsize
gst_tensor_info_get_size (const GstTensorInfo * info);

/**
 * @brief Initialize the tensors info structure
 * @param info tensors info structure to be initialized
 */
extern void
gst_tensors_info_init (GstTensorsInfo * info);

/**
 * @brief Check the tensors info is valid
 * @param info tensors info structure
 * @return TRUE if info is valid
 */
extern gboolean
gst_tensors_info_validate (const GstTensorsInfo * info);

/**
 * @brief Compare tensors info
 * @param TRUE if equal
 */
extern gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2);

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
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
extern gboolean
gst_tensor_config_is_equal (const GstTensorConfig * c1,
    const GstTensorConfig * c2);

/**
 * @brief Parse structure and set tensor config info (for other/tensor)
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_config_from_structure (GstTensorConfig * config,
    const GstStructure * structure);

/**
 * @brief Get tensor caps from tensor config (for other/tensor)
 * @param config tensor config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensor_caps_from_config (const GstTensorConfig * config);

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
 * @brief Compare tensor config info (for other/tensors)
 * @param TRUE if equal
 */
extern gboolean
gst_tensors_config_is_equal (const GstTensorsConfig * c1,
    const GstTensorsConfig * c2);

/**
 * @brief Parse structure and set tensors config (for other/tensors)
 * @param config tensors config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if no error
 */
extern gboolean
gst_tensors_config_from_structure (GstTensorsConfig * config,
    const GstStructure * structure);

/**
 * @brief Get caps from tensors config (for other/tensors)
 * @param config tensors config info
 * @return caps for given config
 */
extern GstCaps *
gst_tensors_caps_from_config (const GstTensorsConfig * config);

/**
 * @brief Determine if we need zero-padding
 * @return 1 if we need to add (or remove) stride per row from the stream data. 0 otherwise.
 */
extern gint
gst_tensor_video_stride_padding_per_row (GstVideoFormat format, gint width);

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
extern tensor_type get_tensor_type (const gchar * typestr);

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
extern int find_key_strv (const gchar ** strv, const gchar * key);

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
 * @brief Make str(xyz) ==> "xyz" with macro expansion
 */
#define str(s) xstr(s)
#define xstr(s) #s

#include <glib/gprintf.h>
#ifdef TIZEN
#include <dlog.h>
#else
#define dlog_print(loglevel, component, ...) \
  do { \
    g_message(__VA_ARGS__); \
  } while (0)
#endif

/**
 * @brief Debug message print. In Tizen, it uses dlog; otherwise,m it uses g_message().
 */
#define debug_print(cond, ...)	\
  do { \
    if ((cond) == TRUE) { \
      dlog_print(DLOG_DEBUG, "nnstreamer", __FILE__ ":" str(__LINE__) " "  __VA_ARGS__); \
    } \
  } while (0)

/**
 * @brief Error message print. In Tizen, it uses dlog; otherwise,m it uses g_message().
 */
#define err_print(...) dlog_print(DLOG_ERROR, "nnstreamer", __VA_ARGS__)

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
extern void gst_tensors_typefind_function (GstTypeFind * tf, gpointer pdata);

#define GST_TENSOR_TYPEFIND_REGISTER(plugin)  do { \
    gst_type_find_register (plugin, "other/tensorsave", \
        GST_RANK_PRIMARY, gst_tensors_typefind_function, "tnsr", \
        gst_caps_new_simple ("other/tensorsave", NULL, NULL), NULL, NULL)); \
    } while (0)

#ifdef SINGLE_BINARY
#define NNSTREAMER_PLUGIN_INIT(name)	\
  gboolean G_PASTE(nnstreamer_export_, name) (GstPlugin * plugin)
#else
#define NNSTREAMER_PLUGIN_INIT(name)	\
  static gboolean G_PASTE(G_PASTE(gst_, name), _plugin_init) (GstPlugin * plugin)
#endif

G_END_DECLS
#endif /* __GST_TENSOR_COMMON_H__ */
