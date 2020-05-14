/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 */
/**
 * @file nnstreamer-capi-util.c
 * @date 10 June 2019
 * @brief NNStreamer/Utilities C-API Wrapper.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>

#include "nnstreamer.h"
#include "nnstreamer-capi-private.h"
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_filter.h"
#include "nnstreamer_conf.h"
#include <tensor_filter/tensor_filter_common.h>

/**
 * @brief Allocates a tensors information handle with default value.
 */
int
ml_tensors_info_create (ml_tensors_info_h * info)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  *info = tensors_info = g_new0 (ml_tensors_info_s, 1);
  if (tensors_info == NULL) {
    ml_loge ("Failed to allocate the tensors info handle.");
    return ML_ERROR_OUT_OF_MEMORY;
  }

  /* init tensors info struct */
  return ml_tensors_info_initialize (tensors_info);
}

/**
 * @brief Allocates a tensors information handle from gst info.
 */
int
ml_tensors_info_create_from_gst (ml_tensors_info_h * ml_info,
    GstTensorsInfo * gst_info)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!ml_info || !gst_info)
    return ML_ERROR_INVALID_PARAMETER;

  *ml_info = tensors_info = g_new0 (ml_tensors_info_s, 1);
  if (tensors_info == NULL) {
    ml_loge ("Failed to allocate the tensors info handle.");
    return ML_ERROR_OUT_OF_MEMORY;
  }

  /* init and copy tensors info from gst struct */
  ml_tensors_info_initialize (tensors_info);
  ml_tensors_info_copy_from_gst (tensors_info, gst_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Frees the given handle of a tensors information.
 */
int
ml_tensors_info_destroy (ml_tensors_info_h info)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  tensors_info = (ml_tensors_info_s *) info;

  if (!tensors_info)
    return ML_ERROR_INVALID_PARAMETER;

  ml_tensors_info_free (tensors_info);
  g_free (tensors_info);

  return ML_ERROR_NONE;
}

/**
 * @brief Initializes the tensors information with default value.
 */
int
ml_tensors_info_initialize (ml_tensors_info_s * info)
{
  guint i, j;

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  info->num_tensors = 0;

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    info->info[i].name = NULL;
    info->info[i].type = ML_TENSOR_TYPE_UNKNOWN;

    for (j = 0; j < ML_TENSOR_RANK_LIMIT; j++) {
      info->info[i].dimension[j] = 0;
    }
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Validates the given tensor info is valid.
 */
static gboolean
ml_tensor_info_validate (const ml_tensor_info_s * info)
{
  guint i;

  if (!info)
    return FALSE;

  if (info->type < 0 || info->type >= ML_TENSOR_TYPE_UNKNOWN)
    return FALSE;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    if (info->dimension[i] == 0)
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Compares the given tensor info.
 */
static gboolean
ml_tensor_info_compare (const ml_tensor_info_s * i1,
    const ml_tensor_info_s * i2)
{
  guint i;

  if (i1 == NULL || i2 == NULL)
    return FALSE;

  if (i1->type != i2->type)
    return FALSE;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    if (i1->dimension[i] != i2->dimension[i])
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Validates the given tensors info is valid.
 */
int
ml_tensors_info_validate (const ml_tensors_info_h info, bool * valid)
{
  ml_tensors_info_s *tensors_info;
  guint i;

  check_feature_state ();

  if (!valid)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (!tensors_info || tensors_info->num_tensors < 1)
    return ML_ERROR_INVALID_PARAMETER;

  /* init false */
  *valid = false;

  for (i = 0; i < tensors_info->num_tensors; i++) {
    if (!ml_tensor_info_validate (&tensors_info->info[i]))
      goto done;
  }

  *valid = true;

done:
  return ML_ERROR_NONE;
}

/**
 * @brief Compares the given tensors information.
 */
int
ml_tensors_info_compare (const ml_tensors_info_h info1,
    const ml_tensors_info_h info2, bool * equal)
{
  ml_tensors_info_s *i1, *i2;
  guint i;

  check_feature_state ();

  if (info1 == NULL || info2 == NULL || equal == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  i1 = (ml_tensors_info_s *) info1;
  i2 = (ml_tensors_info_s *) info2;

  /* init false */
  *equal = false;

  if (i1->num_tensors != i2->num_tensors)
    goto done;

  for (i = 0; i < i1->num_tensors; i++) {
    if (!ml_tensor_info_compare (&i1->info[i], &i2->info[i]))
      goto done;
  }

  *equal = true;

done:
  return ML_ERROR_NONE;
}

/**
 * @brief Sets the number of tensors with given handle of tensors information.
 */
int
ml_tensors_info_set_count (ml_tensors_info_h info, unsigned int count)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info || count > ML_TENSOR_SIZE_LIMIT)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;
  tensors_info->num_tensors = count;

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the number of tensors with given handle of tensors information.
 */
int
ml_tensors_info_get_count (ml_tensors_info_h info, unsigned int *count)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info || !count)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;
  *count = tensors_info->num_tensors;

  return ML_ERROR_NONE;
}

/**
 * @brief Sets the tensor name with given handle of tensors information.
 */
int
ml_tensors_info_set_tensor_name (ml_tensors_info_h info,
    unsigned int index, const char *name)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (tensors_info->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  if (tensors_info->info[index].name) {
    g_free (tensors_info->info[index].name);
    tensors_info->info[index].name = NULL;
  }

  if (name)
    tensors_info->info[index].name = g_strdup (name);

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the tensor name with given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_name (ml_tensors_info_h info,
    unsigned int index, char **name)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info || !name)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (tensors_info->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  *name = g_strdup (tensors_info->info[index].name);

  return ML_ERROR_NONE;
}

/**
 * @brief Sets the tensor type with given handle of tensors information.
 */
int
ml_tensors_info_set_tensor_type (ml_tensors_info_h info,
    unsigned int index, const ml_tensor_type_e type)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  if (type == ML_TENSOR_TYPE_UNKNOWN)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (tensors_info->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info->info[index].type = type;

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the tensor type with given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_type (ml_tensors_info_h info,
    unsigned int index, ml_tensor_type_e * type)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info || !type)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (tensors_info->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  *type = tensors_info->info[index].type;

  return ML_ERROR_NONE;
}

/**
 * @brief Sets the tensor dimension with given handle of tensors information.
 */
int
ml_tensors_info_set_tensor_dimension (ml_tensors_info_h info,
    unsigned int index, const ml_tensor_dimension dimension)
{
  ml_tensors_info_s *tensors_info;
  guint i;

  check_feature_state ();

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (tensors_info->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    tensors_info->info[index].dimension[i] = dimension[i];
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the tensor dimension with given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_dimension (ml_tensors_info_h info,
    unsigned int index, ml_tensor_dimension dimension)
{
  ml_tensors_info_s *tensors_info;
  guint i;

  check_feature_state ();

  if (!info)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  if (tensors_info->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    dimension[i] = tensors_info->info[index].dimension[i];
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the byte size of the given tensor info.
 */
size_t
ml_tensor_info_get_size (const ml_tensor_info_s * info)
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
      ml_loge ("In the given param, tensor type is invalid.");
      return 0;
  }

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    tensor_size *= info->dimension[i];
  }

  return tensor_size;
}

/**
 * @brief Gets the byte size of the given handle of tensors information.
 */
int
ml_tensors_info_get_tensor_size (ml_tensors_info_h info,
    int index, size_t * data_size)
{
  ml_tensors_info_s *tensors_info;

  check_feature_state ();

  if (!info || !data_size)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;

  /* init 0 */
  *data_size = 0;

  if (index < 0) {
    guint i;

    /* get total byte size */
    for (i = 0; i < tensors_info->num_tensors; i++) {
      *data_size += ml_tensor_info_get_size (&tensors_info->info[i]);
    }
  } else {
    if (tensors_info->num_tensors <= index)
      return ML_ERROR_INVALID_PARAMETER;

    *data_size = ml_tensor_info_get_size (&tensors_info->info[index]);
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Frees the tensors info pointer.
 */
void
ml_tensors_info_free (ml_tensors_info_s * info)
{
  gint i;

  if (!info)
    return;

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    if (info->info[i].name) {
      g_free (info->info[i].name);
      info->info[i].name = NULL;
    }
  }

  ml_tensors_info_initialize (info);
}

/**
 * @brief Frees the tensors data pointer.
 */
int
ml_tensors_data_destroy (ml_tensors_data_h data)
{
  ml_tensors_data_s *_data;
  guint i;

  check_feature_state ();

  if (!data)
    return ML_ERROR_INVALID_PARAMETER;

  _data = (ml_tensors_data_s *) data;

  for (i = 0; i < ML_TENSOR_SIZE_LIMIT; i++) {
    if (_data->tensors[i].tensor) {
      g_free (_data->tensors[i].tensor);
      _data->tensors[i].tensor = NULL;
    }
  }

  g_free (_data);
  return ML_ERROR_NONE;
}

/**
 * @brief Allocates a tensor data frame with the given tensors info. (more info in nnstreamer.h)
 * @note Memory for data buffer is not allocated.
 */
int
ml_tensors_data_create_no_alloc (const ml_tensors_info_h info,
    ml_tensors_data_h * data)
{
  ml_tensors_data_s *_data;
  ml_tensors_info_s *_info;
  gint i;

  check_feature_state ();

  if (data == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *data = NULL;

  _data = g_new0 (ml_tensors_data_s, 1);
  if (!_data) {
    ml_loge ("Failed to allocate the tensors data handle.");
    return ML_ERROR_OUT_OF_MEMORY;
  }

  _info = (ml_tensors_info_s *) info;
  if (_info != NULL) {
    _data->num_tensors = _info->num_tensors;
    for (i = 0; i < _data->num_tensors; i++) {
      _data->tensors[i].size = ml_tensor_info_get_size (&_info->info[i]);
      _data->tensors[i].tensor = NULL;
    }
  }

  *data = _data;
  return ML_ERROR_NONE;
}

/**
 * @brief Allocates a tensor data frame with the given tensors info. (more info in nnstreamer.h)
 */
int
ml_tensors_data_create (const ml_tensors_info_h info, ml_tensors_data_h * data)
{
  gint status = ML_ERROR_STREAMS_PIPE;
  ml_tensors_data_s *_data = NULL;
  gint i;

  check_feature_state ();

  if (info == NULL || data == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  status =
      ml_tensors_data_create_no_alloc (info, (ml_tensors_data_h *) & _data);

  if (status != ML_ERROR_NONE) {
    return status;
  }
  if (!_data) {
    return ML_ERROR_STREAMS_PIPE;
  }

  for (i = 0; i < _data->num_tensors; i++) {
    _data->tensors[i].tensor = g_malloc0 (_data->tensors[i].size);
    if (_data->tensors[i].tensor == NULL) {
      status = ML_ERROR_OUT_OF_MEMORY;
      goto failed;
    }
  }

  *data = _data;
  return ML_ERROR_NONE;

failed:
  for (i = 0; i < _data->num_tensors; i++) {
    g_free (_data->tensors[i].tensor);
  }
  g_free (_data);

  ml_loge ("Failed to allocate the memory block.");
  return status;
}

/**
 * @brief Gets a tensor data of given handle.
 */
int
ml_tensors_data_get_tensor_data (ml_tensors_data_h data, unsigned int index,
    void **raw_data, size_t * data_size)
{
  ml_tensors_data_s *_data;

  check_feature_state ();

  if (!data || !raw_data || !data_size)
    return ML_ERROR_INVALID_PARAMETER;

  _data = (ml_tensors_data_s *) data;

  if (_data->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  *raw_data = _data->tensors[index].tensor;
  *data_size = _data->tensors[index].size;

  return ML_ERROR_NONE;
}

/**
 * @brief Copies a tensor data to given handle.
 */
int
ml_tensors_data_set_tensor_data (ml_tensors_data_h data, unsigned int index,
    const void *raw_data, const size_t data_size)
{
  ml_tensors_data_s *_data;

  check_feature_state ();

  if (!data || !raw_data)
    return ML_ERROR_INVALID_PARAMETER;

  _data = (ml_tensors_data_s *) data;

  if (_data->num_tensors <= index)
    return ML_ERROR_INVALID_PARAMETER;

  if (data_size <= 0 || _data->tensors[index].size < data_size)
    return ML_ERROR_INVALID_PARAMETER;

  if (_data->tensors[index].tensor != raw_data)
    memcpy (_data->tensors[index].tensor, raw_data, data_size);
  return ML_ERROR_NONE;
}

/**
 * @brief Copies tensor meta info.
 */
int
ml_tensors_info_clone (ml_tensors_info_h dest, const ml_tensors_info_h src)
{
  ml_tensors_info_s *dest_info, *src_info;
  guint i, j;

  check_feature_state ();

  dest_info = (ml_tensors_info_s *) dest;
  src_info = (ml_tensors_info_s *) src;

  if (!dest_info || !src_info)
    return ML_ERROR_INVALID_PARAMETER;

  ml_tensors_info_initialize (dest_info);

  dest_info->num_tensors = src_info->num_tensors;

  for (i = 0; i < dest_info->num_tensors; i++) {
    dest_info->info[i].name =
        (src_info->info[i].name) ? g_strdup (src_info->info[i].name) : NULL;
    dest_info->info[i].type = src_info->info[i].type;

    for (j = 0; j < ML_TENSOR_RANK_LIMIT; j++)
      dest_info->info[i].dimension[j] = src_info->info[i].dimension[j];
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Copies tensor meta info from gst tensors info.
 */
void
ml_tensors_info_copy_from_gst (ml_tensors_info_s * ml_info,
    const GstTensorsInfo * gst_info)
{
  guint i, j;
  guint max_dim;

  if (!ml_info || !gst_info)
    return;

  ml_tensors_info_initialize (ml_info);
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

    for (; j < ML_TENSOR_RANK_LIMIT; j++) {
      ml_info->info[i].dimension[j] = 1;
    }
  }
}

/**
 * @brief Copies tensor meta info from gst tensors info.
 */
void
ml_tensors_info_copy_from_ml (GstTensorsInfo * gst_info,
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

    for (; j < NNS_TENSOR_RANK_LIMIT; j++) {
      gst_info->info[i].dimension[j] = 1;
    }
  }
}

/**
 * @brief Initializes the GStreamer library. This is internal function.
 */
int
ml_initialize_gstreamer (void)
{
  GError *err = NULL;

  if (!gst_init_check (NULL, NULL, &err)) {
    if (err) {
      ml_loge ("GStreamer has the following error: %s", err->message);
      g_clear_error (&err);
    } else {
      ml_loge ("Cannot initialize GStreamer. Unknown reason.");
    }

    return ML_ERROR_STREAMS_PIPE;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Internal helper function to validate model files.
 */
static int
_ml_validate_model_file (char **model, unsigned int num_models)
{
  guint i;

  if (!model || num_models < 1) {
    ml_loge ("The required param, model is not provided (null).");
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (i = 0; i < num_models; i++) {
    if (!model[i] || !g_file_test (model[i], G_FILE_TEST_IS_REGULAR)) {
      ml_loge ("The given param, model path [%s] is invalid or not given.",
          GST_STR_NULL (model[i]));
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Gets available nn framework using file extension.
 * @return @c 0 on success. Otherwise a negative error value.
 */
static int
_ml_detect_framework (char **model, unsigned int num_models,
    ml_nnfw_type_e * nnfw)
{
  int status = ML_ERROR_NONE;
  gchar *detected;

  detected =
      gst_tensor_filter_framework_auto_detection ((const gchar **) model,
      num_models);
  if (detected == NULL) {
    ml_loge ("The given model has unknown or not supported extension.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (g_ascii_strcasecmp (detected, "nnfw") == 0) {
    *nnfw = ML_NNFW_TYPE_NNFW;
  } else if (g_ascii_strcasecmp (detected, "tensorflow-lite") == 0) {
    *nnfw = ML_NNFW_TYPE_TENSORFLOW_LITE;
  } else if (g_ascii_strcasecmp (detected, "vivante") == 0) {
    *nnfw = ML_NNFW_TYPE_VIVANTE;
  } else if (g_ascii_strcasecmp (detected, "tensorflow") == 0) {
    *nnfw = ML_NNFW_TYPE_TENSORFLOW;
  } else if (g_ascii_strcasecmp (detected, "custom") == 0) {
    *nnfw = ML_NNFW_TYPE_CUSTOM_FILTER;
  } else {
    ml_loge ("The given model has unknown or not supported extension.");
    status = ML_ERROR_NOT_SUPPORTED;
  }

  if (status == ML_ERROR_NONE) {
    ml_logi ("The given model is supposed a %s model.", detected);

    if (!ml_nnfw_is_available (*nnfw, ML_NNFW_HW_ANY)) {
      ml_loge ("%s is not available.", detected);
      status = ML_ERROR_NOT_SUPPORTED;
    }
  }

  g_free (detected);
  return status;
}

/**
 * @brief Validates the nnfw model file.
 * @since_tizen 5.5
 * @param[in] model The path of model file.
 * @param[in/out] nnfw The type of NNFW.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported, or framework to support this model file is unavailable in the environment.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int
ml_validate_model_file (char **model, unsigned int num_models,
    ml_nnfw_type_e * nnfw)
{
  int status = ML_ERROR_NONE;
  gchar *pos;
  gchar **file_ext;
  guint i;

  if ((status = _ml_validate_model_file (model, num_models)) != ML_ERROR_NONE)
    return status;

  if (!nnfw)
    return ML_ERROR_INVALID_PARAMETER;

  /* Check file extention. */
  file_ext = g_malloc0 (sizeof (char *) * (num_models + 1));
  for (i = 0; i < num_models; i++) {
    if ((pos = strrchr (model[i], '.')) == NULL) {
      ml_loge ("The given model [%s] has invalid extension.", model[i]);
      status = ML_ERROR_INVALID_PARAMETER;
      goto done;
    }

    file_ext[i] = g_ascii_strdown (pos, -1);
  }

  /** @todo Make sure num_models is correct for each nnfw type */
  switch (*nnfw) {
    case ML_NNFW_TYPE_ANY:
      status = _ml_detect_framework (model, num_models, nnfw);
      break;
    case ML_NNFW_TYPE_CUSTOM_FILTER:
      if (g_ascii_strcasecmp (file_ext[0], NNSTREAMER_SO_FILE_EXTENSION) != 0) {
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_TENSORFLOW_LITE:
      if (g_ascii_strcasecmp (file_ext[0], ".tflite") != 0) {
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_TENSORFLOW:
      if (g_ascii_strcasecmp (file_ext[0], ".pb") != 0) {
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_NNFW:
    {
      gchar *model_path = NULL;
      gchar *meta = NULL;

      if (g_ascii_strcasecmp (file_ext[0], ".tflite") != 0 &&
          g_ascii_strcasecmp (file_ext[0], ".circle") != 0) {
        status = ML_ERROR_INVALID_PARAMETER;
        break;
      }

      model_path = g_path_get_dirname (model[0]);
      meta = g_build_filename (model_path, "metadata", "MANIFEST", NULL);
      if (!g_file_test (meta, G_FILE_TEST_IS_REGULAR)) {
        ml_loge ("The given model path [%s] is missing metadata.", model_path);
        status = ML_ERROR_INVALID_PARAMETER;
      }

      g_free (model_path);
      g_free (meta);
      break;
    }
    case ML_NNFW_TYPE_VIVANTE:
    {
      if (num_models != 2) {
        ml_loge ("Vivante requires 2 model files.");
        status = ML_ERROR_INVALID_PARAMETER;
        break;
      }

      if ((g_ascii_strcasecmp (file_ext[0], ".so") == 0 &&
              g_ascii_strcasecmp (file_ext[1], ".nb") == 0) ||
          (g_ascii_strcasecmp (file_ext[0], ".nb") == 0 &&
              g_ascii_strcasecmp (file_ext[1], ".so") == 0)) {
        break;
      }

      ml_loge ("The given model [%s,%s] does not appear Vivante model.",
          model[0], model[1]);
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
    case ML_NNFW_TYPE_MVNC:
    case ML_NNFW_TYPE_OPENVINO:
    case ML_NNFW_TYPE_EDGE_TPU:
      /** @todo Need to check method to validate model */
      ml_loge ("Given NNFW is not supported yet.");
      status = ML_ERROR_NOT_SUPPORTED;
      break;
    case ML_NNFW_TYPE_SNAP:
#if !defined(__ANDROID__)
      ml_loge ("SNAP only can be included in Android (arm64-v8a only).");
      status = ML_ERROR_NOT_SUPPORTED;
#endif
      /* SNAP requires multiple files, set supported if model file exists. */
      break;
    case ML_NNFW_TYPE_SNPE:
#if !defined(__ANDROID__)
      ml_loge ("Given framework, SNPE is not supported yet for non Android (arm64-v8a).");
      status = ML_ERROR_NOT_SUPPORTED;
      break;
#endif
      if (g_ascii_strcasecmp (file_ext[0], ".dlc") != 0) {
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_ARMNN:
      if (g_ascii_strcasecmp (file_ext[0], ".caffemodel") != 0 &&
          g_ascii_strcasecmp (file_ext[0], ".tflite") != 0 &&
          g_ascii_strcasecmp (file_ext[0], ".pb") != 0 &&
          g_ascii_strcasecmp (file_ext[0], ".prototxt") != 0) {
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    default:
      status = ML_ERROR_INVALID_PARAMETER;
      break;
  }

done:
  if (status != ML_ERROR_NONE)
    ml_loge ("The given model file is invalid.");

  g_strfreev (file_ext);
  return status;
}

/**
 * @brief Convert c-api based hw to internal representation
 */
static accl_hw
ml_nnfw_to_accl_hw (const ml_nnfw_hw_e hw)
{
  switch (hw) {
    case ML_NNFW_HW_ANY:
      return ACCL_DEFAULT;
    case ML_NNFW_HW_AUTO:
      return ACCL_AUTO;
    case ML_NNFW_HW_CPU:
      return ACCL_CPU;
#if defined(__aarch64__) || defined(__arm__)
    case ML_NNFW_HW_CPU_NEON:
      return ACCL_CPU_NEON;
#else
    case ML_NNFW_HW_CPU_SIMD:
      return ACCL_CPU_SIMD;
#endif
    case ML_NNFW_HW_GPU:
      return ACCL_GPU;
    case ML_NNFW_HW_NPU:
      return ACCL_NPU;
    case ML_NNFW_HW_NPU_MOVIDIUS:
      return ACCL_NPU_MOVIDIUS;
    case ML_NNFW_HW_NPU_EDGE_TPU:
      return ACCL_NPU_EDGE_TPU;
    case ML_NNFW_HW_NPU_VIVANTE:
      return ACCL_NPU_VIVANTE;
    case ML_NNFW_HW_NPU_SR:
      /** @todo how to get srcn npu */
      return ACCL_NPU_SR;
    default:
      return ACCL_AUTO;
  }
}

/**
 * @brief Internal function to convert accelerator as tensor_filter property format.
 * @note returned value must be freed by the caller
 * @note More details on format can be found in gst_tensor_filter_install_properties() in tensor_filter_common.c.
 */
char *
ml_nnfw_to_str_prop (const ml_nnfw_hw_e hw)
{
  const gchar *hw_name;
  const gchar *use_accl = "true:";
  gchar *str_prop = NULL;

  hw_name = get_accl_hw_str (ml_nnfw_to_accl_hw (hw));
  str_prop = g_strdup_printf ("%s%s", use_accl, hw_name);

  return str_prop;
}

/**
 * @brief Internal function to get the sub-plugin name.
 */
const char *
ml_get_nnfw_subplugin_name (ml_nnfw_type_e nnfw)
{
  static const char *nnfw_subplugin_name[] = {
    [ML_NNFW_TYPE_ANY] = "any", /* DO NOT use this name ('any') to get the sub-plugin */
    [ML_NNFW_TYPE_CUSTOM_FILTER] = "custom",
    [ML_NNFW_TYPE_TENSORFLOW_LITE] = "tensorflow-lite",
    [ML_NNFW_TYPE_TENSORFLOW] = "tensorflow",
    [ML_NNFW_TYPE_NNFW] = "nnfw",
    [ML_NNFW_TYPE_MVNC] = "movidius-ncsdk2",
    [ML_NNFW_TYPE_OPENVINO] = "openvino",
    [ML_NNFW_TYPE_VIVANTE] = "vivante",
    [ML_NNFW_TYPE_EDGE_TPU] = "edgetpu",
    [ML_NNFW_TYPE_ARMNN] = "armnn",
    [ML_NNFW_TYPE_SNAP] = "snap",
    [ML_NNFW_TYPE_SNPE] = "snpe",
    NULL
  };

  return nnfw_subplugin_name[nnfw];
}

/**
 * @brief Checks the availability of the given execution environments.
 */
int
ml_check_nnfw_availability (ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw,
    bool * available)
{
  const GstTensorFilterFramework *fw;
  const char *fw_name = NULL;

  check_feature_state ();

  if (!available)
    return ML_ERROR_INVALID_PARAMETER;

  /* init false */
  *available = false;

  if (nnfw == ML_NNFW_TYPE_ANY)
    return ML_ERROR_INVALID_PARAMETER;

  fw_name = ml_get_nnfw_subplugin_name (nnfw);

  if (fw_name) {
    if ((fw = nnstreamer_filter_find (fw_name)) != NULL) {
      if (fw->checkAvailability
          && fw->checkAvailability (ml_nnfw_to_accl_hw (hw)) != 0) {
        ml_logw ("%s is supported but not with the specified hardware.",
            fw_name);
      } else {
        *available = true;
      }
    } else {
      ml_logw ("%s is not supported.", fw_name);
    }
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Checks the availability of the plugin.
 */
int
ml_check_plugin_availability (const char *plugin_name, const char *element_name)
{
  static gboolean list_loaded = FALSE;
  static gchar **restricted_elements = NULL;

  if (!plugin_name || !element_name) {
    ml_loge ("The name is invalid, failed to check the availability.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!list_loaded) {
    gboolean restricted;

    restricted =
        nnsconf_get_custom_value_bool ("element-restriction",
        "enable_element_restriction", FALSE);
    if (restricted) {
      gchar *elements;

      /* check white-list of available plugins */
      elements =
          nnsconf_get_custom_value_string ("element-restriction",
          "restricted_elements");
      if (elements) {
        restricted_elements = g_strsplit_set (elements, " ,;", -1);
        g_free (elements);
      }
    }

    list_loaded = TRUE;
  }

  /* nnstreamer elements */
  if (g_str_equal (plugin_name, "nnstreamer") &&
      g_str_has_prefix (element_name, "tensor_")) {
    return ML_ERROR_NONE;
  }

  if (restricted_elements &&
      find_key_strv ((const gchar **) restricted_elements, element_name) < 0) {
    ml_logw ("The element %s is restricted.", element_name);
    return ML_ERROR_NOT_SUPPORTED;
  }

  return ML_ERROR_NONE;
}
