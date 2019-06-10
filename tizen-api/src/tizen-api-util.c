/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
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
 */
/**
 * @file tizen-api-util.c
 * @date 10 June 2019
 * @brief Tizen NNStreamer/Utilities C-API Wrapper.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <nnstreamer/nnstreamer_plugin_api.h>

#include "nnstreamer.h"
#include "tizen-api-private.h"

static int ml_internal_error_code = ML_ERROR_NONE;

/**
 * @brief Gets the last error code.
 */
int
ml_util_get_last_error (void)
{
  return ml_internal_error_code;
}

/**
 * @brief Sets the last error code.
 */
void
ml_util_set_error (int error_code)
{
  ml_internal_error_code = error_code;
}

/**
 * @brief Initializes the tensors info.
 */
void
ml_util_initialize_tensors_info (ml_tensors_info_s * info)
{
  guint i, j;

  if (!info)
    return;

  info->num_tensors = 0;

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    info->info[i].name = NULL;
    info->info[i].type = ML_TENSOR_TYPE_UNKNOWN;

    for (j = 0; j < ML_TENSOR_RANK_LIMIT; j++) {
      info->info[i].dimension[j] = 0;
    }
  }
}

/**
 * @brief Validates the given tensor info is valid.
 */
int
ml_util_validate_tensor_info (const ml_tensor_info_s * info)
{
  guint i;

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  if (info->type < 0 || info->type >= ML_TENSOR_TYPE_UNKNOWN)
    return ML_ERROR_INVALID_PARAMETER;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    if (info->dimension[i] == 0)
      return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Validates the given tensors info is valid.
 */
int
ml_util_validate_tensors_info (const ml_tensors_info_s * info)
{
  guint i;

  if (!info || info->num_tensors < 1)
    return ML_ERROR_INVALID_PARAMETER;

  for (i = 0; i < info->num_tensors; i++) {
    /* Failed if returned value is not 0 (ML_ERROR_NONE) */
    if (ml_util_validate_tensor_info (&info->info[i]) != ML_ERROR_NONE)
      return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the byte size of the given tensor info.
 */
size_t
ml_util_get_tensor_size (const ml_tensor_info_s * info)
{
  size_t tensor_size;
  gint i;

  if (!info)
    return 0;

  switch (info->type) {
    case ML_TENSOR_TYPE_INT8:
    case ML_TENSOR_TYPE_UINT8:
      tensor_size = 1;
      break;
    case ML_TENSOR_TYPE_INT16:
    case ML_TENSOR_TYPE_UINT16:
      tensor_size = 2;
      break;
    case ML_TENSOR_TYPE_INT32:
    case ML_TENSOR_TYPE_UINT32:
    case ML_TENSOR_TYPE_FLOAT32:
      tensor_size = 4;
      break;
    case ML_TENSOR_TYPE_FLOAT64:
    case ML_TENSOR_TYPE_INT64:
    case ML_TENSOR_TYPE_UINT64:
      tensor_size = 8;
      break;
    default:
      dloge ("In the given param, tensor type is invalid.");
      return 0;
  }

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    tensor_size *= info->dimension[i];
  }

  return tensor_size;
}

/**
 * @brief Gets the byte size of the given tensors info.
 */
size_t
ml_util_get_tensors_size (const ml_tensors_info_s * info)
{
  size_t tensor_size;
  gint i;

  if (!info)
    return 0;

  tensor_size = 0;
  for (i = 0; i < info->num_tensors; i++) {
    tensor_size += ml_util_get_tensor_size (&info->info[i]);
  }

  return tensor_size;
}

/**
 * @brief Frees the tensors info pointer.
 */
void
ml_util_free_tensors_info (ml_tensors_info_s * info)
{
  gint i;

  if (!info)
    return;

  for (i = 0; i < info->num_tensors; i++) {
    if (info->info[i].name) {
      g_free (info->info[i].name);
      info->info[i].name = NULL;
    }
  }

  ml_util_initialize_tensors_info (info);
}

/**
 * @brief Frees the tensors data pointer.
 */
void
ml_util_free_tensors_data (ml_tensors_data_s ** data)
{
  gint i;

  if (data == NULL || (*data) == NULL)
    return;

  for (i = 0; i < (*data)->num_tensors; i++) {
    if ((*data)->tensors[i].tensor) {
      g_free ((*data)->tensors[i].tensor);
      (*data)->tensors[i].tensor = NULL;
    }
  }

  g_free (*data);
  *data = NULL;
}

/**
 * @brief Allocates a tensor data frame with the given tensors info. (more info in nnstreamer.h)
 */
ml_tensors_data_s *
ml_util_allocate_tensors_data (const ml_tensors_info_s * info)
{
  ml_tensors_data_s *data;
  gint i;

  if (!info) {
    ml_util_set_error (ML_ERROR_INVALID_PARAMETER);
    return NULL;
  }

  data = g_new0 (ml_tensors_data_s, 1);
  if (!data) {
    dloge ("Failed to allocate the memory block.");
    ml_util_set_error (ML_ERROR_STREAMS_PIPE);
    return NULL;
  }

  data->num_tensors = info->num_tensors;
  for (i = 0; i < data->num_tensors; i++) {
    data->tensors[i].size = ml_util_get_tensor_size (&info->info[i]);
    data->tensors[i].tensor = g_malloc0 (data->tensors[i].size);
  }

  ml_util_set_error (ML_ERROR_NONE);
  return data;
}

/**
 * @brief Copies tensor meta info.
 */
void
ml_util_copy_tensors_info (ml_tensors_info_s * dest,
    const ml_tensors_info_s * src)
{
  guint i, j;

  if (!dest || !src)
    return;

  ml_util_initialize_tensors_info (dest);

  dest->num_tensors = src->num_tensors;

  for (i = 0; i < dest->num_tensors; i++) {
    dest->info[i].name =
        (src->info[i].name) ? g_strdup (src->info[i].name) : NULL;
    dest->info[i].type = src->info[i].type;

    for (j = 0; j < ML_TENSOR_RANK_LIMIT; j++)
      dest->info[i].dimension[j] = src->info[i].dimension[j];
  }
}

/**
 * @brief Copies tensor meta info from gst tensots info.
 */
void
ml_util_copy_tensors_info_from_gst (ml_tensors_info_s * ml_info,
    const GstTensorsInfo * gst_info)
{
  guint i, j;
  guint max_dim;

  if (!ml_info || !gst_info)
    return;

  ml_util_initialize_tensors_info (ml_info);
  max_dim = MIN (ML_TENSOR_RANK_LIMIT, NNS_TENSOR_RANK_LIMIT);

  ml_info->num_tensors = gst_info->num_tensors;

  for (i = 0; i < gst_info->num_tensors; i++) {
    /* Copy name string */
    if (gst_info->info[i].name) {
      ml_info->info[i].name = g_strdup (gst_info->info[i].name);
    }

    /* Set tensor type */
    switch (gst_info->info[i].type) {
      case _NNS_INT32:
        ml_info->info[i].type = ML_TENSOR_TYPE_INT32;
        break;
      case _NNS_UINT32:
        ml_info->info[i].type = ML_TENSOR_TYPE_UINT32;
        break;
      case _NNS_INT16:
        ml_info->info[i].type = ML_TENSOR_TYPE_INT16;
        break;
      case _NNS_UINT16:
        ml_info->info[i].type = ML_TENSOR_TYPE_UINT16;
        break;
      case _NNS_INT8:
        ml_info->info[i].type = ML_TENSOR_TYPE_INT8;
        break;
      case _NNS_UINT8:
        ml_info->info[i].type = ML_TENSOR_TYPE_UINT8;
        break;
      case _NNS_FLOAT64:
        ml_info->info[i].type = ML_TENSOR_TYPE_FLOAT64;
        break;
      case _NNS_FLOAT32:
        ml_info->info[i].type = ML_TENSOR_TYPE_FLOAT32;
        break;
      case _NNS_INT64:
        ml_info->info[i].type = ML_TENSOR_TYPE_INT64;
        break;
      case _NNS_UINT64:
        ml_info->info[i].type = ML_TENSOR_TYPE_UINT64;
        break;
      default:
        ml_info->info[i].type = ML_TENSOR_TYPE_UNKNOWN;
        break;
    }

    /* Set dimension */
    for (j = 0; j < max_dim; j++) {
      ml_info->info[i].dimension[j] = gst_info->info[i].dimension[j];
    }

    for ( ; j < ML_TENSOR_RANK_LIMIT; j++) {
      ml_info->info[i].dimension[j] = 1;
    }
  }
}

/**
 * @brief Copies tensor meta info from gst tensots info.
 */
void
ml_util_copy_tensors_info_from_ml (GstTensorsInfo * gst_info,
  const ml_tensors_info_s * ml_info)
{
  guint i, j;
  guint max_dim;

  if (!gst_info || !ml_info)
    return;

  gst_tensors_info_init (gst_info);
  max_dim = MIN (ML_TENSOR_RANK_LIMIT, NNS_TENSOR_RANK_LIMIT);

  gst_info->num_tensors = ml_info->num_tensors;

  for (i = 0; i < ml_info->num_tensors; i++) {
    /* Copy name string */
    if (ml_info->info[i].name) {
      gst_info->info[i].name = g_strdup (ml_info->info[i].name);
    }

    /* Set tensor type */
    switch (ml_info->info[i].type) {
      case ML_TENSOR_TYPE_INT32:
        gst_info->info[i].type = _NNS_INT32;
        break;
      case ML_TENSOR_TYPE_UINT32:
        gst_info->info[i].type = _NNS_UINT32;
        break;
      case ML_TENSOR_TYPE_INT16:
        gst_info->info[i].type = _NNS_INT16;
        break;
      case ML_TENSOR_TYPE_UINT16:
        gst_info->info[i].type = _NNS_UINT16;
        break;
      case ML_TENSOR_TYPE_INT8:
        gst_info->info[i].type = _NNS_INT8;
        break;
      case ML_TENSOR_TYPE_UINT8:
        gst_info->info[i].type = _NNS_UINT8;
        break;
      case ML_TENSOR_TYPE_FLOAT64:
        gst_info->info[i].type = _NNS_FLOAT64;
        break;
      case ML_TENSOR_TYPE_FLOAT32:
        gst_info->info[i].type = _NNS_FLOAT32;
        break;
      case ML_TENSOR_TYPE_INT64:
        gst_info->info[i].type = _NNS_INT64;
        break;
      case ML_TENSOR_TYPE_UINT64:
        gst_info->info[i].type = _NNS_UINT64;
        break;
      default:
        gst_info->info[i].type = _NNS_END;
        break;
    }

    /* Set dimension */
    for (j = 0; j < max_dim; j++) {
      gst_info->info[i].dimension[j] = ml_info->info[i].dimension[j];
    }

    for ( ; j < NNS_TENSOR_RANK_LIMIT; j++) {
      gst_info->info[i].dimension[j] = 1;
    }
  }
}

/**
 * @brief Checks the availability of the given execution environments.
 */
int
ml_util_check_nnfw (ml_nnfw_e nnfw, ml_nnfw_hw_e hw)
{
  /** @todo fill this function */
  return ML_ERROR_NONE;
}
