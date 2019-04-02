/**
 * NNStreamer Common Header's Contents
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
 * @file	tensor_common.c
 * @date	29 May 2018
 * @brief	Common data for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <tensor_common.h>
#include <string.h>

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
    return _NNS_TEXT;
  }

  if (g_str_equal (name, "application/octet-stream")) {
    return _NNS_OCTET;
  }

  /* unknown or unsupported type */
  return _NNS_MEDIA_END;
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
 * @brief Compare tensor info
 * @param TRUE if equal
 */
gboolean
gst_tensor_info_is_equal (const GstTensorInfo * i1, const GstTensorInfo * i2)
{
  guint i;

  g_return_val_if_fail (i1 != NULL, FALSE);
  g_return_val_if_fail (i2 != NULL, FALSE);

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

  dest->name = (src->name) ? g_strdup (src->name) : NULL;
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
 * @param TRUE if equal
 */
gboolean
gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2)
{
  guint i;

  g_return_val_if_fail (i1 != NULL, FALSE);
  g_return_val_if_fail (i2 != NULL, FALSE);

  if (i1->num_tensors != i2->num_tensors) {
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
      info->info[i].name = str_name;
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
      g_string_append (types, tensor_element_typename[info->info[i].type]);

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
      g_string_append (names, info->info[i].name);

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

  /** @todo 1/2 == 2/4 Don't say they are different! */
  if (c1->rate_n != c2->rate_n || c1->rate_d != c2->rate_d) {
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
        tensor_element_typename[config->info.type], NULL);
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

  if (c1->rate_n != c2->rate_n || c1->rate_d != c2->rate_d) {
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

  /* remove spaces */
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

/**
 * @brief Find the index value of the given key string array
 * @return Corresponding index. Returns -1 if not found.
 * @param strv Null terminated array of gchar *
 * @param key The key string value
 */
int
find_key_strv (const gchar ** strv, const gchar * key)
{
  int cursor = 0;

  g_assert (strv != NULL);
  while (strv[cursor]) {
    if (!g_ascii_strcasecmp (strv[cursor], key))
      return cursor;
    cursor++;
  }

  return -1;                    /* Not Found */
}

/**
 * @brief A callback for typefind, trying to find whether a file is other/tensors or not.
 * For the concrete definition of headers, please look at the wiki page of nnstreamer:
 * https://github.com/nnsuite/nnstreamer/wiki/Design-External-Save-Format-for-other-tensor-and-other-tensors-Stream-for-TypeFind
 */
void
gst_tensors_typefind_function (GstTypeFind * tf, gpointer pdata)
{
  const guint8 *data = gst_type_find_peek (tf, 0, 40);  /* The first 40 bytes are header-0 in v.1 protocol */
  const char formatstr[] = "TENSORST";
  const unsigned int *supported_version = (const unsigned int *) (&data[8]);
  const unsigned int *num_tensors = (const unsigned int *) (&data[12]);
  if (data &&
      memcmp (data, formatstr, 8) == 0 &&
      *supported_version == 1 && *num_tensors <= 16 && *num_tensors >= 1) {
    gst_type_find_suggest (tf, GST_TYPE_FIND_MAXIMUM,
        gst_caps_new_simple ("other/tensorsave", NULL, NULL));
  }
}

/**
 * @brief A function call to decide current timestamp among collected pads based on PTS.
 * It will decide current timestamp according to sync option.
 */
gboolean
gst_tensor_set_current_time (GstCollectPads * collect,
    GstClockTime * current_time, tensor_time_sync_data sync)
{
  GSList *walk = NULL;
  walk = collect->data;
  gboolean isEOS = FALSE;
  guint count = 0;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    walk = g_slist_next (walk);
    GstBuffer *buf = NULL;
    buf = gst_collect_pads_peek (collect, data);

    if (buf == NULL) {
      isEOS = TRUE;
      return isEOS;
    }

    switch (sync.mode) {
      case SYNC_SLOWEST:
        if (*current_time < GST_BUFFER_PTS (buf))
          *current_time = GST_BUFFER_PTS (buf);
        break;
      case SYNC_BASEPAD:
        if (count == sync.data_basepad.sink_id)
          *current_time = GST_BUFFER_PTS (buf);
        break;
      default:
        break;
    }
    count++;
    gst_buffer_unref (buf);
  }

  return isEOS;
}

/**
 * @brief A function call to make tensors from collected pads.
 * It decide which buffer is going to be used according to sync option.
 */
gboolean
gst_gen_tensors_from_collectpad (GstCollectPads * collect,
    tensor_time_sync_data sync, GstClockTime current_time,
    gboolean * need_buffer, GstBuffer * tensors_buf, GstTensorsConfig * configs)
{
  GSList *walk = NULL;
  GstMemory *mem;
  gboolean isEOS = FALSE;
  gint old_numerator = G_MAXINT;
  gint old_denominator = G_MAXINT;
  gint counting = 0;
  GstTensorsConfig in_configs;
  GstClockTime base = 0;
  guint i = 0;

  walk = collect->data;

  if (sync.mode == SYNC_BASEPAD) {
    walk = g_slist_nth (walk, sync.data_basepad.sink_id);
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorCollectPadData *pad = (GstTensorCollectPadData *) data;
    GstBuffer *buf = NULL;
    buf = gst_collect_pads_peek (collect, data);
    if (buf != NULL) {
      if (pad->buffer != NULL)
        base =
            MIN (sync.data_basepad.duration,
            ABS (GST_CLOCK_DIFF (GST_BUFFER_PTS (buf),
                    GST_BUFFER_PTS (pad->buffer))) - 1);
      gst_buffer_unref (buf);
    }
  }

  walk = collect->data;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorCollectPadData *pad = (GstTensorCollectPadData *) data;
    GstCaps *caps = gst_pad_get_current_caps (pad->pad);
    GstStructure *s = gst_caps_get_structure (caps, 0);

    gst_tensors_config_from_structure (&in_configs, s);
    g_assert (gst_tensors_config_validate (&in_configs));

    if (in_configs.rate_d < old_denominator)
      old_denominator = in_configs.rate_d;
    if (in_configs.rate_n < old_numerator)
      old_numerator = in_configs.rate_n;

    gst_caps_unref (caps);

    walk = g_slist_next (walk);

    GstBuffer *buf = NULL;

    buf = gst_collect_pads_peek (collect, data);

    switch (sync.mode) {
      case SYNC_SLOWEST:
        if (buf != NULL) {
          if (GST_BUFFER_PTS (buf) < current_time) {
            gst_buffer_unref (buf);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = gst_collect_pads_pop (collect, data);
            *need_buffer = TRUE;
            return FALSE;
          }

          if (pad->buffer != NULL &&
              (ABS (GST_CLOCK_DIFF (current_time,
                          GST_BUFFER_PTS (pad->buffer))) <
                  ABS (GST_CLOCK_DIFF (current_time, GST_BUFFER_PTS (buf))))) {
            gst_buffer_unref (buf);
            buf = pad->buffer;
          } else {
            gst_buffer_unref (buf);
            buf = gst_collect_pads_pop (collect, data);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = buf;
          }
        } else {
          buf = pad->buffer;
        }
        break;
      case SYNC_NOSYNC:
        if (buf != NULL) {
          gst_buffer_unref (buf);
          buf = gst_collect_pads_pop (collect, data);
        }
        break;
      case SYNC_BASEPAD:
        if (buf != NULL) {
          if (GST_BUFFER_PTS (buf) < current_time) {
            gst_buffer_unref (buf);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = gst_collect_pads_pop (collect, data);
            *need_buffer = TRUE;
            return FALSE;
          }

          if ((pad->buffer != NULL &&
                  (ABS (GST_CLOCK_DIFF (current_time,
                              GST_BUFFER_PTS (buf))) > base))) {
            gst_buffer_unref (buf);
            buf = pad->buffer;
          } else {
            gst_buffer_unref (buf);
            buf = gst_collect_pads_pop (collect, data);
            if (pad->buffer != NULL)
              gst_buffer_unref (pad->buffer);
            pad->buffer = buf;
          }
        } else {
          buf = pad->buffer;
        }
        break;
      default:
        break;
    }

    if (GST_IS_BUFFER (buf)) {
      guint n_mem = gst_buffer_n_memory (buf);

      g_assert (n_mem == in_configs.info.num_tensors);
      g_assert ((counting + n_mem) < NNS_TENSOR_SIZE_LIMIT);

      for (i = 0; i < n_mem; ++i) {
        mem = gst_buffer_get_memory (buf, i);
        gst_buffer_append_memory (tensors_buf, mem);
        configs->info.info[counting] = in_configs.info.info[i];
        counting++;
      }
      if (sync.mode == SYNC_NOSYNC) {
        gst_buffer_unref (buf);
      }

    } else {
      isEOS = TRUE;
      return isEOS;
    }
  }

  configs->rate_d = old_denominator;
  configs->rate_n = old_numerator;

  GST_BUFFER_PTS (tensors_buf) = current_time;
  return isEOS;
}
