/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file nnstreamer_plugin_api_util_impl.c
 * @date 28 Jan 2022
 * @brief Tensor common util functions for NNStreamer. (No gst dependency)
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>
#include "nnstreamer_plugin_api_util.h"
#include "nnstreamer_log.h"

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
  [_NNS_FLOAT16] = "float16",
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
  [_NNS_FLOAT16] = 2,
  [_NNS_END] = 0,
};

/**
 * @brief String representations for tensor format.
 */
static const gchar *tensor_format_name[] = {
  [_NNS_TENSOR_FORMAT_STATIC] = "static",
  [_NNS_TENSOR_FORMAT_FLEXIBLE] = "flexible",
  [_NNS_TENSOR_FORMAT_SPARSE] = "sparse",
  [_NNS_TENSOR_FORMAT_END] = NULL
};

/**
 * @brief Internal function, copied from gst_util_greatest_common_divisor() to remove dependency of gstreamer.
 */
static gint
_gcd (gint a, gint b)
{
  while (b != 0) {
    int temp = a;

    a = b;
    b = temp % b;
  }

  return ABS (a);
}

/**
 * @brief Internal function, copied from gst_util_fraction_compare() to remove dependency of gstreamer.
 */
static gint
_compare_rate (gint a_n, gint a_d, gint b_n, gint b_d)
{
  gint64 new_num_1;
  gint64 new_num_2;
  gint gcd;

  g_return_val_if_fail (a_d != 0 && b_d != 0, 0);

  /* Simplify */
  gcd = _gcd (a_n, a_d);
  a_n /= gcd;
  a_d /= gcd;

  gcd = _gcd (b_n, b_d);
  b_n /= gcd;
  b_d /= gcd;

  /* fractions are reduced when set, so we can quickly see if they're equal */
  if (a_n == b_n && a_d == b_d)
    return 0;

  /* extend to 64 bits */
  new_num_1 = ((gint64) a_n) * b_d;
  new_num_2 = ((gint64) b_n) * a_d;
  if (new_num_1 < new_num_2)
    return -1;
  if (new_num_1 > new_num_2)
    return 1;

  /* Should not happen because a_d and b_d are not 0 */
  g_return_val_if_reached (0);
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
    nns_logd
        ("Failed to validate tensor info. type: %s. Please specify tensor type. e.g., type=uint8 ",
        _STR_NULL (gst_tensor_get_type_string (info->type)));
    _nnstreamer_error_write
        ("Failed to validate tensor info. type: %s. Please specify tensor type. e.g., type=uint8 ",
        _STR_NULL (gst_tensor_get_type_string (info->type)));
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
  if (!gst_tensor_info_validate (i1) || !gst_tensor_info_validate (i2)) {
    return FALSE;
  }

  if (i1->type != i2->type) {
    nns_logd ("Tensor info is not equal. Given tensor types %s vs %s",
        _STR_NULL (gst_tensor_get_type_string (i1->type)),
        _STR_NULL (gst_tensor_get_type_string (i2->type)));
    return FALSE;
  }

  if (!gst_tensor_dimension_is_equal (i1->dimension, i2->dimension)) {
    g_autofree gchar *_dim1 = gst_tensor_get_dimension_string (i1->dimension);
    g_autofree gchar *_dim2 = gst_tensor_get_dimension_string (i2->dimension);
    nns_logd ("Tensor info is not equal. Given tensor dimensions %s vs %s",
        _dim1, _dim2);
    return FALSE;
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
 * @brief Convert GstTensorInfo structure to GstTensorMetaInfo.
 * @param[in] info GstTensorInfo to be converted
 * @param[out] meta tensor meta structure to be filled
 * @return TRUE if successfully set the meta
 */
gboolean
gst_tensor_info_convert_to_meta (GstTensorInfo * info, GstTensorMetaInfo * meta)
{
  guint i;

  g_return_val_if_fail (gst_tensor_info_validate (info), FALSE);
  g_return_val_if_fail (meta != NULL, FALSE);

  gst_tensor_meta_info_init (meta);

  meta->type = info->type;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    /** @todo handle rank from info.dimension */
    if (info->dimension[i] > 0)
      meta->dimension[i] = info->dimension[i];
    else
      break;
  }

  return TRUE;
}

/**
 * @brief Get tensor rank
 * @param info tensor info structure
 * @return tensor rank (Minimum rank is 1 if given info is valid)
 */
guint
gst_tensor_info_get_rank (const GstTensorInfo * info)
{
  gint idx;

  g_return_val_if_fail (info != NULL, 0);

  /** rank is at least 1 */
  for (idx = NNS_TENSOR_RANK_LIMIT - 1; idx > 0; idx--) {
    if (info->dimension[idx] != 1)
      break;
  }
  /** @todo use gst_tensor_dimension_get_rank (info->dimension) after 0-init dim is done */
  return idx + 1;
}

/**
 * @brief Get the pointer of nth tensor information.
 */
GstTensorInfo *
gst_tensors_info_get_nth_info (GstTensorsInfo * info, guint nth)
{
  g_return_val_if_fail (info != NULL, NULL);

  if (nth < NNS_TENSOR_SIZE_LIMIT)
    return &info->info[nth];

  if (!gst_tensors_info_extra_create (info))
    return NULL;

  if (nth < NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT)
    return &info->extra[nth - NNS_TENSOR_SIZE_LIMIT];

  nns_loge ("Failed to get the information, invalid index %u.", nth);
  return NULL;
}

/**
 * @brief Allocate and initialize the extra info in given tensors info.
 * @param[in,out] info tensors info to be updated.
 */
gboolean
gst_tensors_info_extra_create (GstTensorsInfo * info)
{
  GstTensorInfo *new;
  guint i;

  g_return_val_if_fail (info != NULL, FALSE);

  if (info->extra) {
    return TRUE;
  }

  new = g_try_new0 (GstTensorInfo, NNS_TENSOR_SIZE_EXTRA_LIMIT);
  if (!new) {
    nns_loge ("Failed to allocate memory for extra tensors info");
    return FALSE;
  }

  for (i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; ++i) {
    gst_tensor_info_init (&new[i]);
  }

  info->extra = new;

  return TRUE;
}

/**
 * @brief Free allocated extra info in given tensors info.
 * @param[in,out] info tensors info whose extra info is to be freed.
 */
void
gst_tensors_info_extra_free (GstTensorsInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  if (info->extra) {
    for (i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; ++i)
      gst_tensor_info_free (&info->extra[i]);

    g_free (info->extra);
    info->extra = NULL;
  }
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
  info->extra = NULL;

  /** @note default format is static */
  info->format = _NNS_TENSOR_FORMAT_STATIC;

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

  for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
    gst_tensor_info_free (&info->info[i]);
  }

  if (info->extra)
    gst_tensors_info_extra_free (info);
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
  GstTensorInfo *_info;
  gsize data_size = 0;
  guint i;

  g_return_val_if_fail (info != NULL, 0);
  g_return_val_if_fail (index < (gint) info->num_tensors, 0);

  if (index < 0) {
    for (i = 0; i < info->num_tensors; ++i) {
      _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, i);
      data_size += gst_tensor_info_get_size (_info);
    }
  } else {
    _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, index);
    data_size = gst_tensor_info_get_size (_info);
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
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, FALSE);

  /* tensor stream format */
  if (info->format >= _NNS_TENSOR_FORMAT_END) {
    nns_logd
        ("Failed to validate tensors info, format: %s. format should be one of %s.",
        _STR_NULL (gst_tensor_get_format_string (info->format)),
        GST_TENSOR_FORMAT_ALL);
    _nnstreamer_error_write
        ("Failed to validate tensors info, format: %s. format should be one of %s.",
        _STR_NULL (gst_tensor_get_format_string (info->format)),
        GST_TENSOR_FORMAT_ALL);
    return FALSE;
  }

  /* cannot check tensor info when tensor is not static */
  if (info->format != _NNS_TENSOR_FORMAT_STATIC) {
    return TRUE;
  }

  if (info->num_tensors < 1) {
    nns_logd
        ("Failed to validate tensors info. the number of tensors: %d. the number of tensors should be greater than 0.",
        info->num_tensors);
    _nnstreamer_error_write
        ("Failed to validate tensors info. the number of tensors: %d. the number of tensors should be greater than 0.",
        info->num_tensors);
    return FALSE;
  }

  for (i = 0; i < info->num_tensors; i++) {
    _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, i);

    if (!gst_tensor_info_validate (_info))
      return FALSE;
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
  GstTensorInfo *_info1, *_info2;

  g_return_val_if_fail (i1 != NULL, FALSE);
  g_return_val_if_fail (i2 != NULL, FALSE);

  if (i1->format != i2->format || i1->format == _NNS_TENSOR_FORMAT_END) {
    nns_logd ("Tensors info is not equal. format: %s vs %s ",
        _STR_NULL (gst_tensor_get_format_string (i1->format)),
        _STR_NULL (gst_tensor_get_format_string (i2->format)));
    return FALSE;
  }

  /* cannot compare tensor info when tensor is not static */
  if (i1->format != _NNS_TENSOR_FORMAT_STATIC) {
    return TRUE;
  }

  if (!gst_tensors_info_validate (i1) || !gst_tensors_info_validate (i2)) {
    return FALSE;
  }

  if (i1->num_tensors != i2->num_tensors) {
    nns_logd ("Tensors info is not equal. the number of tensors: %d vs %d. ",
        i1->num_tensors, i2->num_tensors);
    return FALSE;
  }

  for (i = 0; i < i1->num_tensors; i++) {
    _info1 = gst_tensors_info_get_nth_info ((GstTensorsInfo *) i1, i);
    _info2 = gst_tensors_info_get_nth_info ((GstTensorsInfo *) i2, i);

    if (!gst_tensor_info_is_equal (_info1, _info2)) {
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
  guint i, num;
  GstTensorInfo *_dest, *_src;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  gst_tensors_info_init (dest);
  num = dest->num_tensors = src->num_tensors;
  dest->format = src->format;

  if (src->extra != NULL) {
    gst_tensors_info_extra_create (dest);
  }

  for (i = 0; i < num; i++) {
    _dest = gst_tensors_info_get_nth_info (dest, i);
    _src = gst_tensors_info_get_nth_info ((GstTensorsInfo *) src, i);

    gst_tensor_info_copy (_dest, _src);
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
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, 0);

  if (dim_string) {
    guint i;
    gchar **str_dims;

    str_dims = g_strsplit_set (dim_string, ",.", -1);
    num_dims = g_strv_length (str_dims);

    if (num_dims > NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT) {
      nns_logw ("Invalid param, dimensions (%d) max (%d)\n",
          num_dims, NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT);

      num_dims = NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT;
    }

    if (num_dims >= NNS_TENSOR_SIZE_LIMIT)
      gst_tensors_info_extra_create (info);

    for (i = 0; i < num_dims; i++) {
      _info = gst_tensors_info_get_nth_info (info, i);
      gst_tensor_parse_dimension (str_dims[i], _info->dimension);
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
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, 0);

  if (type_string) {
    guint i;
    gchar **str_types;

    str_types = g_strsplit_set (type_string, ",.", -1);
    num_types = g_strv_length (str_types);

    if (num_types > NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT) {
      nns_logw ("Invalid param, types (%d) max (%d)\n",
          num_types, NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT);

      num_types = NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT;
    }

    if (num_types >= NNS_TENSOR_SIZE_LIMIT)
      gst_tensors_info_extra_create (info);

    for (i = 0; i < num_types; i++) {
      _info = gst_tensors_info_get_nth_info (info, i);
      _info->type = gst_tensor_get_type (str_types[i]);
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
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, 0);

  if (name_string) {
    guint i;
    gchar **str_names;

    str_names = g_strsplit (name_string, ",", -1);
    num_names = g_strv_length (str_names);

    if (num_names > NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT) {
      nns_logw ("Invalid param, names (%d) max (%d)\n",
          num_names, NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT);

      num_names = NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT;
    }

    if (num_names >= NNS_TENSOR_SIZE_LIMIT)
      gst_tensors_info_extra_create (info);

    for (i = 0; i < num_names; i++) {
      gchar *str_name;

      _info = gst_tensors_info_get_nth_info (info, i);
      g_free (_info->name);
      _info->name = NULL;

      str_name = g_strstrip (g_strdup (str_names[i]));
      if (str_name && strlen (str_name))
        _info->name = str_name;
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
  return gst_tensors_info_get_rank_dimensions_string (info,
      NNS_TENSOR_RANK_LIMIT);
}

/**
 * @brief Get the string of dimensions in tensors info and rank count
 * @param info tensors info structure
 * @param rank rank count of given tensor dimension
 * @return Formatted string of given dimension
 * @note If rank count is 3, then returned string is 'd1:d2:d3`.
 * The returned value should be freed with g_free()
 */
gchar *
gst_tensors_info_get_rank_dimensions_string (const GstTensorsInfo * info,
    const unsigned int rank)
{
  gchar *dim_str = NULL;
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *dimensions = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, i);
      dim_str = gst_tensor_get_rank_dimension_string (_info->dimension, rank);

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
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *types = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, i);

      if (_info->type != _NNS_END)
        g_string_append (types, gst_tensor_get_type_string (_info->type));

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
  GstTensorInfo *_info;

  g_return_val_if_fail (info != NULL, NULL);

  if (info->num_tensors > 0) {
    guint i;
    GString *names = g_string_new (NULL);

    for (i = 0; i < info->num_tensors; i++) {
      _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, i);

      if (_info->name)
        g_string_append (names, _info->name);

      if (i < info->num_tensors - 1) {
        g_string_append (names, ",");
      }
    }

    name_str = g_string_free (names, FALSE);
  }

  return name_str;
}

/**
 * @brief GstTensorsInfo represented as a string. Caller should free it.
 * @param info GstTensorsInfo structure
 * @return The newly allocated string representing the tensorsinfo. Free after use.
 */
gchar *
gst_tensors_info_to_string (const GstTensorsInfo * info)
{
  GString *gstr = g_string_new (NULL);
  unsigned int i;
  unsigned int limit = info->num_tensors;
  GstTensorInfo *_info;

  g_string_append_printf (gstr, "Num_Tensors = %u, Tensors = [",
      info->num_tensors);
  if (limit > NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT) {
    limit = NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT;
    g_string_append_printf (gstr,
        "(Num_Tensors out of bound. Showing %d only)", limit);
  }

  for (i = 0; i < limit; i++) {
    const gchar *name;
    const gchar *type;
    gchar *dim;

    _info = gst_tensors_info_get_nth_info ((GstTensorsInfo *) info, i);
    name = _info->name;
    type = gst_tensor_get_type_string (_info->type);
    dim = gst_tensor_get_dimension_string (_info->dimension);

    g_string_append_printf (gstr, "{\"%s\", %s, %s}%s",
        name ? name : "", type, dim, (i == info->num_tensors - 1) ? "]" : ", ");

    g_free (dim);
  }

  return g_string_free (gstr, FALSE);
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
 * @brief Free allocated data in tensors config structure
 * @param config tensors config structure
 */
void
gst_tensors_config_free (GstTensorsConfig * config)
{
  g_return_if_fail (config != NULL);

  gst_tensors_info_free (&config->info);
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
    nns_logd
        ("Failed to validate tensors config. framerate: %d/%d. framerate should be numerator >= 0 and denominator > 0.",
        config->rate_n, config->rate_d);
    _nnstreamer_error_write
        ("Failed to validate tensors config. framerate: %d/%d. framerate should be numerator >= 0 and denominator > 0.",
        config->rate_n, config->rate_d);
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

  if (!gst_tensors_config_validate (c1) || !gst_tensors_config_validate (c2)) {
    return FALSE;
  }

  if (_compare_rate (c1->rate_n, c1->rate_d, c2->rate_n, c2->rate_d)) {
    nns_logd ("Tensors config is not equal. framerate: %d/%d vs %d/%d.",
        c1->rate_n, c1->rate_d, c2->rate_n, c2->rate_d);
    return FALSE;
  }

  return gst_tensors_info_is_equal (&c1->info, &c2->info);
}

/**
 * @brief Copy tensors config
 */
void
gst_tensors_config_copy (GstTensorsConfig * dest, const GstTensorsConfig * src)
{
  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  gst_tensors_info_copy (&dest->info, &src->info);
  dest->rate_n = src->rate_n;
  dest->rate_d = src->rate_d;
}

/**
 * @brief Tensor config represented as a string. Caller should free it.
 * @param config tensor config structure
 * @return The newly allocated string representing the config. Free after use.
 */
gchar *
gst_tensors_config_to_string (const GstTensorsConfig * config)
{
  GString *gstr = g_string_new (NULL);
  const gchar *fmt = gst_tensor_get_format_string (config->info.format);
  g_string_append_printf (gstr, "Format = %s, Framerate = %d/%d",
      fmt, config->rate_n, config->rate_d);
  if (config->info.format == _NNS_TENSOR_FORMAT_STATIC) {
    gchar *infostr = gst_tensors_info_to_string (&config->info);
    g_string_append_printf (gstr, ", %s", infostr);
    g_free (infostr);
  }
  return g_string_free (gstr, FALSE);
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
  gboolean is_valid = FALSE;

  i = gst_tensor_dimension_get_rank (dim);
  if (i == 0)
    goto done;

  for (; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (dim[i] > 0)
      goto done;
  }

  is_valid = TRUE;

done:
  if (!is_valid) {
    nns_logd
        ("Failed to validate tensor dimension. The dimension string should be in the form of d1:...:d8, d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1. Here, dN is a positive integer.");
    _nnstreamer_error_write
        ("Failed to validate tensor dimension. The dimension string should be in the form of d1:...:d8, d1:d2:d3:d4, d1:d2:d3, d1:d2, or d1. Here, dN is a positive integer.");
  }

  return is_valid;
}

/**
 * @brief Compare the tensor dimension.
 * @return TRUE if given tensors have same dimension.
 */
gboolean
gst_tensor_dimension_is_equal (const tensor_dim dim1, const tensor_dim dim2)
{
  guint i;

  /* Do not compare invalid dimensions. */
  if (!gst_tensor_dimension_is_valid (dim1) ||
      !gst_tensor_dimension_is_valid (dim2))
    return FALSE;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (dim1[i] != dim2[i]) {
      /* Supposed dimension is same if remained dimension is 1. */
      if (dim1[i] > 1 || dim2[i] > 1)
        return FALSE;
    }
  }

  return TRUE;
}

/**
 * @brief Get the rank of tensor dimension.
 * @param dim tensor dimension.
 * @return tensor rank (Minimum rank is 1 if given dimension is valid)
 */
guint
gst_tensor_dimension_get_rank (const tensor_dim dim)
{
  guint i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (dim[i] == 0)
      break;
  }

  return i;
}

/**
 * @brief Parse tensor dimension parameter string
 * @return The Rank. 0 if error.
 * @param dimstr The dimension string in the format of d1:...:d8, d1:d2:d3, d1:d2, or d1, where dN is a positive integer and d1 is the innermost dimension; i.e., dim[d8][d7][d6][d5][d4][d3][d2][d1];
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

  /* 0-init */
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    dim[i] = 0;

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

  /**
   * @todo remove below lines
   * (0-initialized before parsing the string, filled remained dimension with 0)
   */
  for (; i < NNS_TENSOR_RANK_LIMIT; i++)
    dim[i] = 1;

  g_strfreev (strv);
  g_free (dim_string);
  return rank;
}

/**
 * @brief Get dimension string from given tensor dimension.
 * @param dim tensor dimension
 * @return Formatted string of given dimension (d1:d2:d3:d4:d5:d6:d7:d8).
 * @note The returned value should be freed with g_free()
 */
gchar *
gst_tensor_get_dimension_string (const tensor_dim dim)
{
  return gst_tensor_get_rank_dimension_string (dim, NNS_TENSOR_RANK_LIMIT);
}

/**
 * @brief Get dimension string from given tensor dimension and rank count.
 * @param dim tensor dimension
 * @param rank rank count of given tensor dimension
 * @return Formatted string of given dimension
 * @note If rank count is 3, then returned string is 'd1:d2:d3`.
 * The returned value should be freed with g_free().
 */
gchar *
gst_tensor_get_rank_dimension_string (const tensor_dim dim,
    const unsigned int rank)
{
  guint i;
  GString *dim_str;
  guint actual_rank;

  dim_str = g_string_new (NULL);

  if (rank == 0 || rank > NNS_TENSOR_RANK_LIMIT)
    actual_rank = NNS_TENSOR_RANK_LIMIT;
  else
    actual_rank = rank;

  for (i = 0; i < actual_rank; i++) {
    if (dim[i] == 0)
      break;

    g_string_append_printf (dim_str, "%u", dim[i]);

    if (i < actual_rank - 1 && dim[i + 1] > 0) {
      g_string_append (dim_str, ":");
    }
  }

  return g_string_free (dim_str, FALSE);
}

/**
 * @brief Compare dimension strings
 * @return TRUE if equal, FALSE if given dimension strings are invalid or not equal.
 */
gboolean
gst_tensor_dimension_string_is_equal (const gchar * dimstr1,
    const gchar * dimstr2)
{
  tensor_dim dim1, dim2;
  guint rank1, rank2, i, j, num_tensors1, num_tensors2;
  gchar **strv1;
  gchar **strv2;
  gboolean is_equal = FALSE;

  strv1 = g_strsplit_set (dimstr1, ",.", -1);
  strv2 = g_strsplit_set (dimstr2, ",.", -1);

  num_tensors1 = g_strv_length (strv1);
  num_tensors2 = g_strv_length (strv2);

  if (num_tensors1 != num_tensors2)
    goto done;

  for (i = 0; i < num_tensors1; i++) {
    for (j = 0; j < NNS_TENSOR_RANK_LIMIT; j++)
      dim1[j] = dim2[j] = 0;

    rank1 = gst_tensor_parse_dimension (strv1[i], dim1);
    rank2 = gst_tensor_parse_dimension (strv2[i], dim2);

    /* 'rank 0' means invalid dimension */
    if (!rank1 || !rank2 || !gst_tensor_dimension_is_equal (dim1, dim2))
      goto done;
  }

  /* Compared all tensor dimensions from input string. */
  is_equal = TRUE;

done:
  g_strfreev (strv1);
  g_strfreev (strv2);

  return is_equal;
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
    if (dim[i] == 0)
      break;

    count *= dim[i];
  }

  return (i > 0) ? count : 0;
}

/**
 * @brief Get element size of tensor type (byte per element)
 */
gsize
gst_tensor_get_element_size (tensor_type type)
{
  g_return_val_if_fail (type >= 0 && type <= _NNS_END, 0);

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
  } else if (g_regex_match_simple ("^float(16|32|64)$",
          type_string, G_REGEX_CASELESS, 0)) {
    size = (gsize) g_ascii_strtoull (&type_string[5], NULL, 10);

    switch (size) {
      case 16:
        type = _NNS_FLOAT16;
        break;
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
  g_return_val_if_fail (type >= 0 && type <= _NNS_END, NULL);

  return tensor_element_typename[type];
}

/**
 * @brief Get tensor format from string input.
 * @param format_str The string format name, supposed to be one of tensor_format_name[].
 * @return Corresponding tensor_format. _NNS_TENSOR_FORMAT_END if unrecognized value is there.
 */
tensor_format
gst_tensor_get_format (const gchar * format_str)
{
  gint idx;
  tensor_format format = _NNS_TENSOR_FORMAT_END;

  idx = find_key_strv (tensor_format_name, format_str);
  if (idx >= 0)
    format = (tensor_format) idx;

  return format;
}

/**
 * @brief Get tensor format string.
 */
const gchar *
gst_tensor_get_format_string (tensor_format format)
{
  g_return_val_if_fail (format >= 0 && format <= _NNS_TENSOR_FORMAT_END, NULL);

  return tensor_format_name[format];
}

/**
 * @brief Magic number of tensor meta.
 */
#define GST_TENSOR_META_MAGIC (0xfeedcced)

/**
 * @brief Macro to check the tensor meta.
 */
#define GST_TENSOR_META_MAGIC_VALID(m) ((m) == GST_TENSOR_META_MAGIC)

/**
 * @brief Macro to check the meta version.
 */
#define GST_TENSOR_META_VERSION_VALID(v) (((v) & 0xDE000000) == 0xDE000000)

/**
 * @brief Macro to get the version of tensor meta.
 */
#define GST_TENSOR_META_MAKE_VERSION(major,minor) ((major) << 12 | (minor) | 0xDE000000)

/**
 * @brief The version of tensor meta.
 */
#define GST_TENSOR_META_VERSION GST_TENSOR_META_MAKE_VERSION(1,0)

/**
 * @brief Macro to check the version of tensor meta.
 */
#define GST_TENSOR_META_IS_V1(v) (GST_TENSOR_META_VERSION_VALID(v) && (((v) & 0x00FFF000) & GST_TENSOR_META_MAKE_VERSION(1,0)))

/**
 * @brief Macro to check the meta is valid.
 */
#define GST_TENSOR_META_IS_VALID(m) ((m) && GST_TENSOR_META_MAGIC_VALID ((m)->magic) && GST_TENSOR_META_VERSION_VALID ((m)->version))

/**
 * @brief Initialize the tensor meta info structure.
 * @param[in,out] meta tensor meta structure to be initialized
 */
void
gst_tensor_meta_info_init (GstTensorMetaInfo * meta)
{
  g_return_if_fail (meta != NULL);

  /* zero-init */
  memset (meta, 0, sizeof (GstTensorMetaInfo));

  meta->magic = GST_TENSOR_META_MAGIC;
  meta->version = GST_TENSOR_META_VERSION;
  meta->type = _NNS_END;
  meta->format = _NNS_TENSOR_FORMAT_STATIC;
  meta->media_type = _NNS_TENSOR;
}

/**
 * @brief Get the version of tensor meta.
 * @param[in] meta tensor meta structure
 * @param[out] major pointer to get the major version number
 * @param[out] minor pointer to get the minor version number
 */
void
gst_tensor_meta_info_get_version (GstTensorMetaInfo * meta,
    guint * major, guint * minor)
{
  g_return_if_fail (meta != NULL);

  if (!GST_TENSOR_META_IS_VALID (meta))
    return;

  if (major)
    *major = (meta->version & 0x00FFF000) >> 12;

  if (minor)
    *minor = (meta->version & 0x00000FFF);
}

/**
 * @brief Check the meta info is valid.
 * @param[in] meta tensor meta structure
 * @return TRUE if given meta is valid
 */
gboolean
gst_tensor_meta_info_validate (GstTensorMetaInfo * meta)
{
  guint i;

  g_return_val_if_fail (meta != NULL, FALSE);

  if (!GST_TENSOR_META_IS_VALID (meta))
    return FALSE;

  if (meta->type >= _NNS_END) {
    nns_logd ("Failed to validate tensor meta info. type: %s. ",
        _STR_NULL (gst_tensor_get_type_string (meta->type)));
    return FALSE;
  }

  for (i = 0; i < NNS_TENSOR_META_RANK_LIMIT; i++) {
    if (meta->dimension[i] == 0) {
      if (i == 0) {
        gchar *dim_str = gst_tensor_get_dimension_string (meta->dimension);
        nns_logd ("Failed to validate tensor meta info. Given dimension: %s",
            dim_str);
        g_free (dim_str);
        return FALSE;
      }
      break;
    }
  }

  if (meta->format >= _NNS_TENSOR_FORMAT_END) {
    nns_logd ("Failed to validate tensors meta info. format: %s. ",
        _STR_NULL (gst_tensor_get_format_string (meta->format)));
    return FALSE;
  }

  if (meta->media_type > _NNS_TENSOR) {
    nns_logd ("Failed to validate tensor meta info. invalid media type: %d.",
        meta->media_type);
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Get the header size to handle a tensor meta.
 * @param[in] meta tensor meta structure
 * @return Header size for meta info (0 if meta is invalid)
 */
gsize
gst_tensor_meta_info_get_header_size (GstTensorMetaInfo * meta)
{
  g_return_val_if_fail (meta != NULL, 0);

  if (!GST_TENSOR_META_IS_VALID (meta))
    return 0;

  /* return fixed size for meta version */
  if (GST_TENSOR_META_IS_V1 (meta->version)) {
    return 128;
  }

  return 0;
}

/**
 * @brief Get the data size calculated from tensor meta.
 * @param[in] meta tensor meta structure
 * @return The data size for meta info (0 if meta is invalid)
 */
gsize
gst_tensor_meta_info_get_data_size (GstTensorMetaInfo * meta)
{
  guint i;
  gsize dsize;

  g_return_val_if_fail (meta != NULL, 0);

  if (!GST_TENSOR_META_IS_VALID (meta))
    return 0;

  dsize = gst_tensor_get_element_size (meta->type);

  if (meta->format == _NNS_TENSOR_FORMAT_SPARSE) {
    return meta->sparse_info.nnz * (dsize + sizeof (guint));
  }

  for (i = 0; i < NNS_TENSOR_META_RANK_LIMIT; i++) {
    if (meta->dimension[i] == 0)
      break;

    dsize *= meta->dimension[i];
  }

  return (i > 0) ? dsize : 0;
}

/**
 * @brief Update header from tensor meta.
 * @param[in] meta tensor meta structure
 * @param[out] header pointer to header to be updated
 * @return TRUE if successfully set the header
 * @note User should allocate enough memory for header (see gst_tensor_meta_info_get_header_size()).
 */
gboolean
gst_tensor_meta_info_update_header (GstTensorMetaInfo * meta, gpointer header)
{
  gsize hsize;

  g_return_val_if_fail (header != NULL, FALSE);
  g_return_val_if_fail (gst_tensor_meta_info_validate (meta), FALSE);

  hsize = gst_tensor_meta_info_get_header_size (meta);

  memset (header, 0, hsize);

  memcpy (header, meta, sizeof (GstTensorMetaInfo));
  return TRUE;
}

/**
 * @brief Parse header and fill the tensor meta.
 * @param[out] meta tensor meta structure to be filled
 * @param[in] header pointer to header to be parsed
 * @return TRUE if successfully set the meta
 */
gboolean
gst_tensor_meta_info_parse_header (GstTensorMetaInfo * meta, gpointer header)
{
  uint32_t *val = (uint32_t *) header;

  g_return_val_if_fail (header != NULL, FALSE);
  g_return_val_if_fail (meta != NULL, FALSE);

  gst_tensor_meta_info_init (meta);

  meta->magic = val[0];
  meta->version = val[1];
  meta->type = val[2];
  memcpy (meta->dimension, &val[3],
      sizeof (uint32_t) * NNS_TENSOR_META_RANK_LIMIT);
  meta->format = val[19];
  meta->media_type = val[20];

  switch ((tensor_format) meta->format) {
    case _NNS_TENSOR_FORMAT_SPARSE:
      meta->sparse_info.nnz = val[21];
      break;
    default:
      break;
  }

  /** @todo update meta info for each version */
  return gst_tensor_meta_info_validate (meta);
}

/**
 * @brief Convert GstTensorMetaInfo structure to GstTensorInfo.
 * @param[in] meta tensor meta structure to be converted
 * @param[out] info GstTensorInfo to be filled
 * @return TRUE if successfully set the info
 */
gboolean
gst_tensor_meta_info_convert (GstTensorMetaInfo * meta, GstTensorInfo * info)
{
  guint i;

  g_return_val_if_fail (info != NULL, FALSE);
  g_return_val_if_fail (gst_tensor_meta_info_validate (meta), FALSE);

  gst_tensor_info_init (info);

  info->type = meta->type;

  for (i = 0; i < NNS_TENSOR_META_RANK_LIMIT; i++) {    /* lgtm[cpp/constant-comparison] */
    if (i >= NNS_TENSOR_RANK_LIMIT) {
      if (meta->dimension[i] > 0) {
        nns_loge ("Given meta has invalid dimension (dimension[%u] %u).",
            i, meta->dimension[i]);
        nns_loge ("Failed to set info, max rank should be %u.",
            NNS_TENSOR_RANK_LIMIT);
        return FALSE;
      }

      /* tensor-info max rank is NNS_TENSOR_RANK_LIMIT */
      break;
    }

    /** @todo handle rank from info.dimension (Fill 0, not 1) */
    info->dimension[i] = (meta->dimension[i] > 0) ? meta->dimension[i] : 1;
  }

  return TRUE;
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
  while (strv[cursor] && key) {
    if (g_ascii_strcasecmp (strv[cursor], key) == 0)
      return cursor;
    cursor++;
  }

  return -1;                    /* Not Found */
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
