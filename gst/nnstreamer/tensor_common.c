/**
 * NNStreamer Common Header's Contents
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
 * @file	tensor_common.c
 * @date	29 May 2018
 * @brief	Common data for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <tensor_common.h>
#include <string.h>

/**
 * @brief String representations for each tensor element type.
 */
static const gchar *tensor_element_typename[] = {
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
 * @brief Byte-per-element of each tensor element type.
 */
static const guint tensor_element_size[] = {
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
 * @brief Get media type from structure
 * @param structure structure to be interpreted
 * @return corresponding media type (returns _NNS_MEDIA_INVALID for unsupported type)
 */
media_type
gst_tensor_media_type_from_structure (const GstStructure * structure)
{
  const gchar *name;

  name = gst_structure_get_name (structure);

  g_return_val_if_fail (name != NULL, _NNS_MEDIA_INVALID);

  if (g_str_has_prefix (name, "video/")) {
    return _NNS_VIDEO;
  }

  if (g_str_has_prefix (name, "audio/")) {
    return _NNS_AUDIO;
  }

  if (g_str_has_prefix (name, "text/")) {
    return _NNS_TEXT;
  }

  if (g_str_equal (name, "application/octet-stream")) {
    return _NNS_OCTET;
  }

  /* unknown or unsupported type */
  return _NNS_MEDIA_INVALID;
}

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
 * @brief Free allocated data in tensor info structure
 * @param info tensor info structure
 */
void
gst_tensor_info_free (GstTensorInfo * info)
{
  g_return_if_fail (info != NULL);

  if (info->name) {
    g_free (info->name);
    info->name = NULL;
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
      gst_tensor_get_element_size (info->type);

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
 * @brief Compare tensor info
 * @return TRUE if equal, FALSE if given tensor infos are invalid or not equal.
 */
gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2)
{
  guint i;

  if (gst_tensor_info_validate (i1) == FALSE ||
      gst_tensor_info_validate (i2) == FALSE) {
    return FALSE;
  }

  if (i1->type != i2->type) {
    return FALSE;
  }

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (i1->dimension[i] != i2->dimension[i]) {
      return FALSE;
    }
  }

  /* matched all */
  return TRUE;
}

/**
 * @brief Copy tensor info up to n elements
 * @note Copied info should be freed with gst_tensor_info_free()
 */
void
gst_tensor_info_copy_n (GstTensorInfo * dest, const GstTensorInfo * src,
    const guint n)
{
  guint i;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  dest->name = g_strdup (src->name);
  dest->type = src->type;

  for (i = 0; i < n; i++) {
    dest->dimension[i] = src->dimension[i];
  }
}

/**
 * @brief Copy tensor info
 * @note Copied info should be freed with gst_tensor_info_free()
 */
void
gst_tensor_info_copy (GstTensorInfo * dest, const GstTensorInfo * src)
{
  gst_tensor_info_copy_n (dest, src, NNS_TENSOR_RANK_LIMIT);
}

/**
 * @brief Get tensor rank
 * @param info tensor info structure
 * @return tensor rank (Minimum rank is 1 if given info is valid)
 */
gint
gst_tensor_info_get_rank (const GstTensorInfo * info)
{
  gint idx;

  g_return_val_if_fail (info != NULL, 0);

  /** rank is at least 1 */
  for (idx = NNS_TENSOR_RANK_LIMIT - 1; idx > 0; idx--) {
    if (info->dimension[idx] != 1)
      break;
  }

  return idx + 1;
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
 * @brief Free allocated data in tensors info structure
 * @param info tensors info structure
 */
void
gst_tensors_info_free (GstTensorsInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  for (i = 0; i < info->num_tensors; i++) {
    gst_tensor_info_free (&info->info[i]);
  }
}

/**
 * @brief Get data size of single tensor
 * @param info tensors info structure
 * @param index the index of tensor (-1 to get total size of tensors)
 * @return data size
 */
gsize
gst_tensors_info_get_size (const GstTensorsInfo * info, gint index)
{
  gsize data_size = 0;
  guint i;

  g_return_val_if_fail (info != NULL, 0);
  g_return_val_if_fail (index < (gint) info->num_tensors, 0);

  if (index < 0) {
    for (i = 0; i < info->num_tensors; ++i)
      data_size += gst_tensor_info_get_size (&info->info[i]);
  } else {
    data_size = gst_tensor_info_get_size (&info->info[index]);
  }

  return data_size;
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
 * @brief Compare tensors info
 * @return TRUE if equal, FALSE if given tensor infos are invalid or not equal.
 */
gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2)
{
  guint i;

  g_return_val_if_fail (i1 != NULL, FALSE);
  g_return_val_if_fail (i2 != NULL, FALSE);

  if (i1->num_tensors != i2->num_tensors || i1->num_tensors < 1) {
    return FALSE;
  }

  for (i = 0; i < i1->num_tensors; i++) {
    if (!gst_tensor_info_is_equal (&i1->info[i], &i2->info[i])) {
      return FALSE;
    }
  }

  /* matched all */
  return TRUE;
}

/**
 * @brief Copy tensor info
 * @note Copied info should be freed with gst_tensors_info_free()
 */
void
gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src)
{
  guint i;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  dest->num_tensors = src->num_tensors;

  for (i = 0; i < src->num_tensors; i++) {
    gst_tensor_info_copy (&dest->info[i], &src->info[i]);
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

    str_dims = g_strsplit_set (dim_string, ",.", -1);
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

    str_types = g_strsplit_set (type_string, ",.", -1);
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
 * @brief Parse the string of names
 * @param info tensors info structure
 * @param name_string string of names
 * @return number of parsed names
 */
guint
gst_tensors_info_parse_names_string (GstTensorsInfo * info,
    const gchar * name_string)
{
  guint num_names = 0;

  g_return_val_if_fail (info != NULL, 0);

  if (name_string) {
    guint i;
    gchar *str_name;
    gchar **str_names;

    str_names = g_strsplit (name_string, ",", -1);
    num_names = g_strv_length (str_names);

    if (num_names > NNS_TENSOR_SIZE_LIMIT) {
      GST_WARNING ("Invalid param, names (%d) max (%d)\n",
          num_names, NNS_TENSOR_SIZE_LIMIT);

      num_names = NNS_TENSOR_SIZE_LIMIT;
    }

    for (i = 0; i < num_names; i++) {
      str_name = g_strdup (str_names[i]);
      g_strstrip (str_name);

      g_free (info->info[i].name);
      info->info[i].name = NULL;

      if (str_name && strlen (str_name))
        info->info[i].name = str_name;
      else
        g_free (str_name);
    }

    g_strfreev (str_names);
  }

  return num_names;
}

/**
 * @brief Get the string of dimensions in tensors info
 * @param info tensors info structure
 * @return string of dimensions in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_dimensions_string (const GstTensorsInfo * info)
{
  gchar *dim_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *dimensions = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      dim_str = gst_tensor_get_dimension_string (info->info[i].dimension);

      g_string_append (dimensions, dim_str);

      if (i < info->num_tensors - 1) {
        g_string_append (dimensions, ",");
      }

      g_free (dim_str);
    }

    dim_str = g_string_free (dimensions, FALSE);
  }

  return dim_str;
}

/**
 * @brief Get the string of types in tensors info
 * @param info tensors info structure
 * @return string of types in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_types_string (const GstTensorsInfo * info)
{
  gchar *type_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *types = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      g_string_append (types, gst_tensor_get_type_string (info->info[i].type));

      if (i < info->num_tensors - 1) {
        g_string_append (types, ",");
      }
    }

    type_str = g_string_free (types, FALSE);
  }

  return type_str;
}

/**
 * @brief Get the string of tensor names in tensors info
 * @param info tensors info structure
 * @return string of names in tensors info (NULL if the number of tensors is 0)
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_names_string (const GstTensorsInfo * info)
{
  gchar *name_str = NULL;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *names = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      if (info->info[i].name) {
        g_string_append (names, info->info[i].name);
      }

      if (i < info->num_tensors - 1) {
        g_string_append (names, ",");
      }
    }

    name_str = g_string_free (names, FALSE);
  }

  return name_str;
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
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
gboolean
gst_tensor_config_is_equal (const GstTensorConfig * c1,
    const GstTensorConfig * c2)
{
  g_return_val_if_fail (c1 != NULL, FALSE);
  g_return_val_if_fail (c2 != NULL, FALSE);

  if (c1->rate_d == 0 || c2->rate_d == 0) {
    return FALSE;
  }

  if (gst_util_fraction_compare (c1->rate_n, c1->rate_d, c2->rate_n,
          c2->rate_d) != 0) {
    return FALSE;
  }

  return gst_tensor_info_is_equal (&c1->info, &c2->info);
}

/**
 * @brief Parse structure and set tensor config info
 * @param config tensor config structure to be filled
 * @param structure structure to be interpreted
 * @return TRUE if ok
 */
gboolean
gst_tensor_config_from_structure (GstTensorConfig * config,
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
 * @brief Get tensor caps from tensor config
 * @param config tensor config info
 * @return caps for given config
 */
GstCaps *
gst_tensor_caps_from_config (const GstTensorConfig * config)
{
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT);

  if (gst_tensor_dimension_is_valid (config->info.dimension)) {
    gchar *dim_str = gst_tensor_get_dimension_string (config->info.dimension);

    gst_caps_set_simple (caps, "dimension", G_TYPE_STRING, dim_str, NULL);
    g_free (dim_str);
  }

  if (config->info.type != _NNS_END) {
    gst_caps_set_simple (caps, "type", G_TYPE_STRING,
        gst_tensor_get_type_string (config->info.type), NULL);
  }

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  return gst_caps_simplify (caps);
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
 * @brief Compare tensor config info
 * @param TRUE if equal
 */
gboolean
gst_tensors_config_is_equal (const GstTensorsConfig * c1,
    const GstTensorsConfig * c2)
{
  g_return_val_if_fail (c1 != NULL, FALSE);
  g_return_val_if_fail (c2 != NULL, FALSE);

  if (c1->rate_d == 0 || c2->rate_d == 0) {
    return FALSE;
  }

  if (gst_util_fraction_compare (c1->rate_n, c1->rate_d, c2->rate_n,
          c2->rate_d) != 0) {
    return FALSE;
  }

  return gst_tensors_info_is_equal (&c1->info, &c2->info);
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

    gst_tensor_config_from_structure (&c, structure);

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
 * @brief Get caps from tensors config (for other/tensors)
 * @param config tensors config info
 * @return caps for given config
 */
GstCaps *
gst_tensors_caps_from_config (const GstTensorsConfig * config)
{
  GstCaps *caps;

  g_return_val_if_fail (config != NULL, NULL);

  caps = gst_caps_from_string (GST_TENSORS_CAP_DEFAULT);

  if (config->info.num_tensors > 0) {
    gchar *dim_str, *type_str;

    dim_str = gst_tensors_info_get_dimensions_string (&config->info);
    type_str = gst_tensors_info_get_types_string (&config->info);

    gst_caps_set_simple (caps, "num_tensors", G_TYPE_INT,
        config->info.num_tensors, NULL);
    gst_caps_set_simple (caps, "dimensions", G_TYPE_STRING, dim_str, NULL);
    gst_caps_set_simple (caps, "types", G_TYPE_STRING, type_str, NULL);

    g_free (dim_str);
    g_free (type_str);
  }

  if (config->rate_n >= 0 && config->rate_d > 0) {
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION,
        config->rate_n, config->rate_d, NULL);
  }

  return gst_caps_simplify (caps);
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

  /* remove spaces */
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
 * @brief Count the number of elements of a tensor
 * @return The number of elements. 0 if error.
 * @param dim The tensor dimension
 */
gulong
gst_tensor_get_element_count (const tensor_dim dim)
{
  gulong count = 1;
  guint i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    count *= dim[i];
  }

  return count;
}

/**
 * @brief Get element size of tensor type (byte per element)
 */
gsize
gst_tensor_get_element_size (tensor_type type)
{
  return tensor_element_size[type];
}

/**
 * @brief Get tensor type from string input.
 * @return Corresponding tensor_type. _NNS_END if unrecognized value is there.
 * @param typestr The string type name, supposed to be one of tensor_element_typename[]
 */
tensor_type
gst_tensor_get_type (const gchar * typestr)
{
  gsize size, len;
  gchar *type_string;
  tensor_type type = _NNS_END;

  if (typestr == NULL)
    return _NNS_END;

  /* remove spaces */
  type_string = g_strdup (typestr);
  g_strstrip (type_string);

  len = strlen (type_string);

  if (len == 0) {
    g_free (type_string);
    return _NNS_END;
  }

  if (g_regex_match_simple ("^uint(8|16|32|64)$",
          type_string, G_REGEX_CASELESS, 0)) {
    size = (gsize) g_ascii_strtoull (&type_string[4], NULL, 10);

    switch (size) {
      case 8:
        type = _NNS_UINT8;
        break;
      case 16:
        type = _NNS_UINT16;
        break;
      case 32:
        type = _NNS_UINT32;
        break;
      case 64:
        type = _NNS_UINT64;
    }
  } else if (g_regex_match_simple ("^int(8|16|32|64)$",
          type_string, G_REGEX_CASELESS, 0)) {
    size = (gsize) g_ascii_strtoull (&type_string[3], NULL, 10);

    switch (size) {
      case 8:
        type = _NNS_INT8;
        break;
      case 16:
        type = _NNS_INT16;
        break;
      case 32:
        type = _NNS_INT32;
        break;
      case 64:
        type = _NNS_INT64;
    }
  } else if (g_regex_match_simple ("^float(32|64)$",
          type_string, G_REGEX_CASELESS, 0)) {
    size = (gsize) g_ascii_strtoull (&type_string[5], NULL, 10);

    switch (size) {
      case 32:
        type = _NNS_FLOAT32;
        break;
      case 64:
        type = _NNS_FLOAT64;
    }
  }

  g_free (type_string);
  return type;
}

/**
 * @brief Get type string of tensor type.
 */
const gchar *
gst_tensor_get_type_string (tensor_type type)
{
  return tensor_element_typename[type];
}

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index. Returns -1 if not found.
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
gint
find_key_strv (const gchar ** strv, const gchar * key)
{
  gint cursor = 0;

  if (strv == NULL) {
    ml_logf_stacktrace
        ("find_key_strv is called with a null pointer. Possible internal logic errors.\n");
    return -1;
  }
  while (strv[cursor]) {
    if (g_ascii_strcasecmp (strv[cursor], key) == 0)
      return cursor;
    cursor++;
  }

  return -1;                    /* Not Found */
}

/**
 * @brief Replaces string.
 * This function deallocates the input source string.
 * @param[in] source The input string. This will be freed when returning the replaced string.
 * @param[in] what The string to search for.
 * @param[in] to The string to be replaced.
 * @param[in] delimiters The characters which specify the place to split the string. Set NULL to replace all matched string.
 * @param[out] count The count of replaced. Set NULL if it is unnecessary.
 * @return Newly allocated string. The returned string should be freed with g_free().
 */
gchar *
replace_string (gchar * source, const gchar * what, const gchar * to,
    const gchar * delimiters, guint * count)
{
  GString *builder;
  gchar *start, *pos, *result;
  guint changed = 0;
  gsize len;

  g_return_val_if_fail (source, NULL);
  g_return_val_if_fail (what && to, source);

  len = strlen (what);
  start = source;

  builder = g_string_new (NULL);
  while ((pos = g_strstr_len (start, -1, what)) != NULL) {
    gboolean skip = FALSE;

    if (delimiters) {
      const gchar *s;
      gchar *prev, *next;
      gboolean prev_split, next_split;

      prev = next = NULL;
      prev_split = next_split = FALSE;

      if (pos != source)
        prev = pos - 1;
      if (*(pos + len) != '\0')
        next = pos + len;

      for (s = delimiters; *s != '\0'; ++s) {
        if (!prev || *s == *prev)
          prev_split = TRUE;
        if (!next || *s == *next)
          next_split = TRUE;
        if (prev_split && next_split)
          break;
      }

      if (!prev_split || !next_split)
        skip = TRUE;
    }

    builder = g_string_append_len (builder, start, pos - start);

    /* replace string if found */
    if (skip)
      builder = g_string_append_len (builder, pos, len);
    else
      builder = g_string_append (builder, to);

    start = pos + len;
    if (!skip)
      changed++;
  }

  /* append remains */
  builder = g_string_append (builder, start);
  result = g_string_free (builder, FALSE);

  if (count)
    *count = changed;

  g_free (source);
  return result;
}

/**
 * @brief Get the version of NNStreamer (string).
 * @return Newly allocated string. The returned string should be freed with g_free().
 */
gchar *
nnstreamer_version_string (void)
{
  gchar *version;

  version = g_strdup_printf ("NNStreamer %s", VERSION);
  return version;
}

/**
 * @brief Get the version of NNStreamer (int, divided).
 * @param[out] major MAJOR.minor.micro, won't set if it's null.
 * @param[out] minor major.MINOR.micro, won't set if it's null.
 * @param[out] micro major.minor.MICRO, won't set if it's null.
 */
void
nnstreamer_version_fetch (guint * major, guint * minor, guint * micro)
{
  if (major)
    *major = (NNSTREAMER_VERSION_MAJOR);
  if (minor)
    *minor = (NNSTREAMER_VERSION_MINOR);
  if (micro)
    *micro = (NNSTREAMER_VERSION_MICRO);
}
