/**
 * NNStreamer Common Plug-In API's Implementation Body
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
 * @brief Implementation body of NNStreamer plug-in APIs
 * @see https://github.com/nnsuite/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com> and Wook Song <wook16.song@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */

#include <string.h>

#include "nnstreamer_plugin_api.h"
#include "tensor_typedef.h"

/**
 * @brief String representations for each tensor element type.
 */
const gchar *tensor_element_typename[] = {
  [_NNS_INT32] = "int32",
  [_NNS_UINT32] = "uint32",
  [_NNS_INT16] = "int16",
  [_NNS_UINT16] = "uint16",
  [_NNS_INT8] = "int8",
  [_NNS_UINT8] = "uint8",
  [_NNS_FLOAT64] = "float64",
  [_NNS_FLOAT32] = "float32",
  [_NNS_INT64] = "int64",
  [_NNS_UINT64] = "uint64",
  [_NNS_END] = NULL,
};

/**
 * @brief Initialize the tensor info structure
 * @param info tensor info structure to be initialized
 */
void
gst_tensor_info_init (GstTensorInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  info->name = NULL;
  info->type = _NNS_END;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    info->dimension[i] = 0;
  }
}

/**
 * @brief Get data size of single tensor
 * @param info tensor info structure
 * @return data size
 */
gsize
gst_tensor_info_get_size (const GstTensorInfo * info)
{
  gsize data_size;

  g_return_val_if_fail (info != NULL, 0);

  data_size = gst_tensor_get_element_count (info->dimension) *
      tensor_element_size[info->type];

  return data_size;
}

/**
 * @brief Check the tensor info is valid
 * @param info tensor info structure
 * @return TRUE if info is valid
 */
gboolean
gst_tensor_info_validate (const GstTensorInfo * info)
{
  g_return_val_if_fail (info != NULL, FALSE);

  if (info->type == _NNS_END) {
    return FALSE;
  }

  /* validate tensor dimension */
  return gst_tensor_dimension_is_valid (info->dimension);
}

/**
 * @brief Initialize the tensors info structure
 * @param info tensors info structure to be initialized
 */
void
gst_tensors_info_init (GstTensorsInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  info->num_tensors = 0;

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    gst_tensor_info_init (&info->info[i]);
  }
}

/**
 * @brief Parse the string of dimensions
 * @param info tensors info structure
 * @param dim_string string of dimensions
 * @return number of parsed dimensions
 */
guint
gst_tensors_info_parse_dimensions_string (GstTensorsInfo * info,
    const gchar * dim_string)
{
  guint num_dims = 0;

  g_return_val_if_fail (info != NULL, 0);

  if (dim_string) {
    guint i;
    gchar **str_dims;

    str_dims = g_strsplit (dim_string, ",", -1);
    num_dims = g_strv_length (str_dims);

    if (num_dims > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, dimensions (%d) max (%d)\n",
          num_dims, NNS_TENSOR_SIZE_LIMIT);

      num_dims = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_dims; i++) {
      gst_tensor_parse_dimension (str_dims[i], info->info[i].dimension);
    }

    g_strfreev (str_dims);
  }

  return num_dims;
}

/**
 * @brief Parse the string of types
 * @param info tensors info structure
 * @param type_string string of types
 * @return number of parsed types
 */
guint
gst_tensors_info_parse_types_string (GstTensorsInfo * info,
    const gchar * type_string)
{
  guint num_types = 0;

  g_return_val_if_fail (info != NULL, 0);

  if (type_string) {
    guint i;
    gchar **str_types;

    str_types = g_strsplit (type_string, ",", -1);
    num_types = g_strv_length (str_types);

    if (num_types > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, types (%d) max (%d)\n",
          num_types, NNS_TENSOR_SIZE_LIMIT);

      num_types = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_types; i++) {
      info->info[i].type = gst_tensor_get_type (str_types[i]);
    }

    g_strfreev (str_types);
  }

  return num_types;
}

/**
 * @brief Check the tensors info is valid
 * @param info tensors info structure
 * @return TRUE if info is valid
 */
gboolean
gst_tensors_info_validate (const GstTensorsInfo * info)
{
  guint i;

  g_return_val_if_fail (info != NULL, FALSE);

  if (info->num_tensors < 1) {
    return FALSE;
  }

  for (i = 0; i < info->num_tensors; i++) {
    if (!gst_tensor_info_validate (&info->info[i])) {
      return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief Initialize the tensor config info structure
 * @param config tensor config structure to be initialized
 */
void
gst_tensor_config_init (GstTensorConfig * config)
{
  g_return_if_fail (config != NULL);

  gst_tensor_info_init (&config->info);

  config->rate_n = -1;
  config->rate_d = -1;
}

/**
 * @brief Check the tensor is all configured
 * @param config tensor config structure
 * @return TRUE if configured
 */
gboolean
gst_tensor_config_validate (const GstTensorConfig * config)
{
  g_return_val_if_fail (config != NULL, FALSE);

  /* framerate (numerator >= 0 and denominator > 0) */
  if (config->rate_n < 0 || config->rate_d <= 0) {
    return FALSE;
  }

  return gst_tensor_info_validate (&config->info);
}

/**
 * @brief Parse structure and set tensor config info (internal static function)
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if ok
 */
static gboolean
gst_tensor_config_from_tensor_info (GstTensorConfig * config,
    const GstStructure * structure)
{
  GstTensorInfo *info;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);
  info = &config->info;

  g_return_val_if_fail (structure != NULL, FALSE);

  if (!gst_structure_has_name (structure, "other/tensor")) {
    const gchar *name = gst_structure_get_name (structure);
    GST_WARNING ("caps is not tensor [%s]\n", name ? name : "Unknown");
    return FALSE;
  }

  if (gst_structure_has_field (structure, "dimension")) {
    const gchar *dim_str = gst_structure_get_string (structure, "dimension");
    gst_tensor_parse_dimension (dim_str, info->dimension);
  }

  if (gst_structure_has_field (structure, "type")) {
    const gchar *type_str = gst_structure_get_string (structure, "type");
    info->type = gst_tensor_get_type (type_str);
  }

  gst_structure_get_fraction (structure, "framerate", &config->rate_n,
      &config->rate_d);

  return TRUE;
}

/**
 * @brief Set the tensor config structure from video info (internal static function)
 * @param config tensor config structure to be filled
 * @param structure caps structure
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_config_from_video_info (GstTensorConfig * config,
    const GstStructure * structure)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/image
   * A 4-D uint8 or float32 Tensor of shape [batch_size, height, width, channels]
   * where channels is 1, 3, or 4.
   */
  const gchar *format_string;
  GstVideoFormat format = GST_VIDEO_FORMAT_UNKNOWN;
  gint width = 0;
  gint height = 0;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  format_string = gst_structure_get_string (structure, "format");
  if (format_string) {
    format = gst_video_format_from_string (format_string);
  }

  gst_structure_get_int (structure, "width", &width);
  gst_structure_get_int (structure, "height", &height);
  gst_structure_get_fraction (structure, "framerate", &config->rate_n,
      &config->rate_d);

  /** [color-space][width][height][frames] */
  switch (format) {
    case GST_VIDEO_FORMAT_GRAY8:
      config->info.type = _NNS_UINT8;
      config->info.dimension[0] = 1;
      break;
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
      config->info.type = _NNS_UINT8;
      config->info.dimension[0] = 3;
      break;
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_xBGR:
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_ARGB:
    case GST_VIDEO_FORMAT_ABGR:
      config->info.type = _NNS_UINT8;
      config->info.dimension[0] = 4;
      break;
    default:
      /** unsupported format */
      GST_WARNING ("Unsupported format = %s\n",
          format_string ? format_string : "Unknown");
      break;
  }

  config->info.dimension[1] = width;
  config->info.dimension[2] = height;
  config->info.dimension[3] = 1; /** Supposed 1 frame in tensor, change this if tensor contains N frames. */
#if 0
  /**
   * @todo To fix coverity issue, now block these lines.
   * If NNS_TENSOR_RANK_LIMIT is larger than 4, unblock these to initialize the tensor dimension.
   */
  gint i;
  for (i = 4; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.dimension[i] = 1;
  }
#endif
  return (config->info.type != _NNS_END);
}

/**
 * @brief Set the tensor config structure from audio info (internal static function)
 * @param config tensor config structure to be filled
 * @param structure caps structure
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_config_from_audio_info (GstTensorConfig * config,
    const GstStructure * structure)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/audio
   * A 3-D float32 Tensor of shape [batch_size, frames, channels]
   * or a 2-D float32 Tensor of shape [batch_size, frames].
   */
  const gchar *format_string;
  GstAudioFormat format = GST_AUDIO_FORMAT_UNKNOWN;
  gint channels = 0;
  gint rate = 0;
  gint i;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  format_string = gst_structure_get_string (structure, "format");
  if (format_string) {
    format = gst_audio_format_from_string (format_string);
  }

  gst_structure_get_int (structure, "channels", &channels);
  gst_structure_get_int (structure, "rate", &rate);

  /** [channels][frames] */
  switch (format) {
    case GST_AUDIO_FORMAT_S8:
      config->info.type = _NNS_INT8;
      break;
    case GST_AUDIO_FORMAT_U8:
      config->info.type = _NNS_UINT8;
      break;
    case GST_AUDIO_FORMAT_S16:
      config->info.type = _NNS_INT16;
      break;
    case GST_AUDIO_FORMAT_U16:
      config->info.type = _NNS_UINT16;
      break;
    case GST_AUDIO_FORMAT_S32:
      config->info.type = _NNS_INT32;
      break;
    case GST_AUDIO_FORMAT_U32:
      config->info.type = _NNS_UINT32;
      break;
    case GST_AUDIO_FORMAT_F32:
      config->info.type = _NNS_FLOAT32;
      break;
    case GST_AUDIO_FORMAT_F64:
      config->info.type = _NNS_FLOAT64;
      break;
    default:
      /** unsupported format */
      GST_WARNING ("Unsupported format = %s\n",
          format_string ? format_string : "Unknown");
      break;
  }

  config->info.dimension[0] = channels;
  config->info.dimension[1] = 1; /** Supposed 1 frame in tensor, change this if tensor contains N frames */

  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.dimension[i] = 1;
  }

  if (rate > 0) {
    config->rate_n = rate;
    config->rate_d = 1;
  }

  return (config->info.type != _NNS_END);
}

/**
 * @brief Set the tensor config structure from text info (internal static function)
 * @param config tensor config structure to be filled
 * @param structure caps structure
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_config_from_text_info (GstTensorConfig * config,
    const GstStructure * structure)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/text
   * A string-type Tensor
   */
  const gchar *format_string;
  gint i;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  format_string = gst_structure_get_string (structure, "format");
  if (format_string) {
    if (g_str_equal (format_string, "utf8")) {
      config->info.type = _NNS_UINT8;
    } else {
      /** unsupported format */
      GST_WARNING ("Unsupported format = %s\n", format_string);
    }
  }

  /** [size][frames] */
  config->info.dimension[0] = GST_TENSOR_STRING_SIZE; /** fixed size of string */
  config->info.dimension[1] = 1; /** Supposed 1 frame in tensor, change this if tensor contains N frames */

  for (i = 2; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.dimension[i] = 1;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /** cannot get the framerate for text type */
    config->rate_n = 0;
    config->rate_d = 1;
  }

  return (config->info.type != _NNS_END);
}

/**
 * @brief Set the tensor config structure from octet stream (internal static function)
 * @param config tensor config structure to be filled
 * @param structure caps structure
 * @note Change tensor dimention and type.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_config_from_octet_stream_info (GstTensorConfig * config,
    const GstStructure * structure)
{
  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  /**
   * Raw byte-stream (application/octet-stream)
   * We cannot get the exact tensor info from caps.
   * All tensor info should be updated.
   */
  config->info.type = _NNS_UINT8;

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /** cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }

  return (config->info.type != _NNS_END);
}

/**
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_END for unsupported type)
 */
media_type
gst_tensor_media_type_from_structure (const GstStructure * structure)
{
  const gchar *name;

  name = gst_structure_get_name (structure);

  g_return_val_if_fail (name != NULL, _NNS_MEDIA_END);

  if (g_str_has_prefix (name, "video/")) {
    return _NNS_VIDEO;
  }

  if (g_str_has_prefix (name, "audio/")) {
    return _NNS_AUDIO;
  }

  if (g_str_has_prefix (name, "text/")) {
    return _NNS_STRING;
  }

  if (g_str_equal (name, "application/octet-stream")) {
    return _NNS_OCTET;
  }

  /** unknown, or not-supported type */
  return _NNS_MEDIA_END;
}

/**
 * @brief Parse structure and set tensor config info
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if no error
 */
gboolean
gst_tensor_config_from_structure (GstTensorConfig * config,
    const GstStructure * structure)
{
  media_type m_type;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  /** update config from tensor stream */
  if (gst_structure_has_name (structure, "other/tensor")) {
    return gst_tensor_config_from_tensor_info (config, structure);
  }

  /** update config from media stream */
  m_type = gst_tensor_media_type_from_structure (structure);

  switch (m_type) {
    case _NNS_VIDEO:
      gst_tensor_config_from_video_info (config, structure);
      break;
    case _NNS_AUDIO:
      gst_tensor_config_from_audio_info (config, structure);
      break;
    case _NNS_STRING:
      gst_tensor_config_from_text_info (config, structure);
      break;
    case _NNS_OCTET:
      gst_tensor_config_from_octet_stream_info (config, structure);
      break;
    default:
      GST_WARNING ("Unsupported type %d\n", m_type);
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Initialize the tensors config info structure (for other/tensors)
 * @param config tensors config structure to be initialized
 */
void
gst_tensors_config_init (GstTensorsConfig * config)
{
  g_return_if_fail (config != NULL);

  gst_tensors_info_init (&config->info);

  config->rate_n = -1;
  config->rate_d = -1;
}

/**
 * @brief Parse structure and set tensors config (for other/tensors)
 * @param config tensors config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if no error
 */
gboolean
gst_tensors_config_from_structure (GstTensorsConfig * config,
    const GstStructure * structure)
{
  const gchar *name;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensors_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  name = gst_structure_get_name (structure);

  if (g_str_equal (name, "other/tensor")) {
    GstTensorConfig c;

    gst_tensor_config_from_tensor_info (&c, structure);

    config->info.num_tensors = 1;
    config->info.info[0] = c.info;
    config->rate_d = c.rate_d;
    config->rate_n = c.rate_n;
  } else if (g_str_equal (name, "other/tensors")) {
    gst_structure_get_int (structure, "num_tensors",
        (gint *) (&config->info.num_tensors));
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);

    if (config->info.num_tensors > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, max size is %d", NNS_TENSOR_SIZE_LIMIT);
      config->info.num_tensors = NNS_TENSOR_SIZE_LIMIT;
    }

    /* parse dimensions */
    if (gst_structure_has_field (structure, "dimensions")) {
      const gchar *dims_str;
      guint num_dims;

      dims_str = gst_structure_get_string (structure, "dimensions");
      num_dims =
          gst_tensors_info_parse_dimensions_string (&config->info, dims_str);

      if (config->info.num_tensors != num_dims) {
        GST_WARNING ("Invalid param, dimensions (%d) tensors (%d)\n",
            num_dims, config->info.num_tensors);
      }
    }

    /* parse types */
    if (gst_structure_has_field (structure, "types")) {
      const gchar *types_str;
      guint num_types;

      types_str = gst_structure_get_string (structure, "types");
      num_types =
          gst_tensors_info_parse_types_string (&config->info, types_str);

      if (config->info.num_tensors != num_types) {
        GST_WARNING ("Invalid param, types (%d) tensors (%d)\n",
            num_types, config->info.num_tensors);
      }
    }
  } else {
    GST_WARNING ("Unsupported type = %s\n", name ? name : "Unknown");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Check the tensors are all configured
 * @param config tensor config structure
 * @return TRUE if configured
 */
gboolean
gst_tensors_config_validate (const GstTensorsConfig * config)
{
  g_return_val_if_fail (config != NULL, FALSE);

  /* framerate (numerator >= 0 and denominator > 0) */
  if (config->rate_n < 0 || config->rate_d <= 0) {
    return FALSE;
  }

  return gst_tensors_info_validate (&config->info);
}

/**
 * @brief Check the tensor dimension is valid
 * @param dim tensor dimension
 * @return TRUE if dimension is valid
 */
gboolean
gst_tensor_dimension_is_valid (const tensor_dim dim)
{
  guint i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    if (dim[i] == 0) {
      return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param dimstr The dimension string in the format of d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d4][d3][d2][d1];
 * @param dim dimension to be filled.
 */
guint
gst_tensor_parse_dimension (const gchar * dimstr, tensor_dim dim)
{
  guint rank = 0;
  guint64 val;
  gchar **strv;
  gchar *dim_string;
  guint i, num_dims;

  if (dimstr == NULL)
    return 0;

  /** remove spaces */
  dim_string = g_strdup (dimstr);
  g_strstrip (dim_string);

  strv = g_strsplit (dim_string, ":", NNS_TENSOR_RANK_LIMIT);
  num_dims = g_strv_length (strv);

  for (i = 0; i < num_dims; i++) {
    g_strstrip (strv[i]);
    if (strv[i] == NULL || strlen (strv[i]) == 0)
      break;

    val = g_ascii_strtoull (strv[i], NULL, 10);
    dim[i] = (uint32_t) val;
    rank = i + 1;
  }

  for (; i < NNS_TENSOR_RANK_LIMIT; i++)
    dim[i] = 1;

  g_strfreev (strv);
  g_free (dim_string);
  return rank;
}

/**
 * @brief Get dimension string from given tensor dimension.
 * @param dim tensor dimension
 * @return Formatted string of given dimension (d1:d2:d3:d4).
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensor_get_dimension_string (const tensor_dim dim)
{
  guint i;
  GString *dim_str;

  dim_str = g_string_new (NULL);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    g_string_append_printf (dim_str, "%d", dim[i]);

    if (i < NNS_TENSOR_RANK_LIMIT - 1) {
      g_string_append (dim_str, ":");
    }
  }

  return g_string_free (dim_str, FALSE);
}

/**
 * @brief Count the number of elemnts of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
gsize
gst_tensor_get_element_count (const tensor_dim dim)
{
  gsize count = 1;
  guint i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    count *= dim[i];
  }

  return count;
}

/**
 * @brief Get tensor_type from string tensor_type input
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
tensor_type
gst_tensor_get_type (const gchar * typestr)
{
  guint len;
  gchar *type_string;
  tensor_type type = _NNS_END;

  if (typestr == NULL)
    return _NNS_END;

  /** remove spaces */
  type_string = g_strdup (typestr);
  g_strstrip (type_string);

  len = strlen (type_string);

  if (type_string[0] == 'u' || type_string[0] == 'U') {
    /**
     * Let's believe the developer and the following three letters are "int"
     * (case insensitive)
     */
    if (len == 6) {             /* uint16, uint32 */
      if (type_string[4] == '1' && type_string[5] == '6')
        type = _NNS_UINT16;
      else if (type_string[4] == '3' && type_string[5] == '2')
        type = _NNS_UINT32;
      else if (type_string[4] == '6' && type_string[5] == '4')
        type = _NNS_UINT64;
    } else if (len == 5) {      /* uint8 */
      if (type_string[4] == '8')
        type = _NNS_UINT8;
    }
  } else if (type_string[0] == 'i' || type_string[0] == 'I') {
    /**
     * Let's believe the developer and the following two letters are "nt"
     * (case insensitive)
     */
    if (len == 5) {             /* int16, int32 */
      if (type_string[3] == '1' && type_string[4] == '6')
        type = _NNS_INT16;
      else if (type_string[3] == '3' && type_string[4] == '2')
        type = _NNS_INT32;
      else if (type_string[3] == '6' && type_string[4] == '4')
        type = _NNS_INT64;
    } else if (len == 4) {      /* int8 */
      if (type_string[3] == '8')
        type = _NNS_INT8;
    }
  } else if (type_string[0] == 'f' || type_string[0] == 'F') {
    /* Let's assume that the following 4 letters are "loat" */
    if (len == 7) {
      if (type_string[5] == '6' && type_string[6] == '4')
        type = _NNS_FLOAT64;
      else if (type_string[5] == '3' && type_string[6] == '2')
        type = _NNS_FLOAT32;
    }
  }

  g_free (type_string);
  return type;
}
