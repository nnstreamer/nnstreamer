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
 * @file nnstreamer-capi-util.c
 * @date 10 June 2019
 * @brief NNStreamer/Utilities C-API Wrapper.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>

#include "nnstreamer.h"
#include "nnstreamer-capi-private.h"
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_filter.h"
#include "nnstreamer_conf.h"

#if defined(__TIZEN__)
  #include <system_info.h>
#endif

/**
 * @brief Internal struct to control tizen feature support (machine_learning.inference).
 * -1: Not checked yet, 0: Not supported, 1: Supported
 */
typedef struct
{
  GMutex mutex;
  int feature_status;
} feature_info_s;

static feature_info_s *feature_info = NULL;

/**
 * @brief Internal function to initialize feature status.
 */
static void
ml_initialize_feature_status (void)
{
  if (feature_info == NULL) {
    feature_info = g_new0 (feature_info_s, 1);
    g_assert (feature_info);

    g_mutex_init (&feature_info->mutex);
    feature_info->feature_status = -1;
  }
}

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
  ml_tensors_info_initialize (tensors_info);

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
static int
ml_tensor_info_validate (const ml_tensor_info_s * info)
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
    /* Failed if returned value is not 0 (ML_ERROR_NONE) */
    if (ml_tensor_info_validate (&tensors_info->info[i]) != ML_ERROR_NONE)
      goto done;
  }

  *valid = true;

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

  *name = tensors_info->info[index].name;

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
    int index, size_t *data_size)
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
 */
int
ml_tensors_data_create (const ml_tensors_info_h info,
    ml_tensors_data_h * data)
{
  ml_tensors_data_s *_data;
  ml_tensors_info_s *tensors_info;
  gint i;

  check_feature_state ();

  if (!info || !data)
    return ML_ERROR_INVALID_PARAMETER;

  tensors_info = (ml_tensors_info_s *) info;
  *data = NULL;

  _data = g_new0 (ml_tensors_data_s, 1);
  if (!_data) {
    ml_loge ("Failed to allocate the memory block.");
    return ML_ERROR_STREAMS_PIPE;
  }

  _data->num_tensors = tensors_info->num_tensors;
  for (i = 0; i < _data->num_tensors; i++) {
    _data->tensors[i].size = ml_tensor_info_get_size (&tensors_info->info[i]);
    _data->tensors[i].tensor = g_malloc0 (_data->tensors[i].size);
    if (_data->tensors[i].tensor == NULL)
      goto failed;
  }

  *data = _data;
  return ML_ERROR_NONE;

failed:
  if (_data) {
    for (i = 0; i < _data->num_tensors; i++) {
      g_free (_data->tensors[i].tensor);
    }
  }

  ml_loge ("Failed to allocate the memory block.");
  return ML_ERROR_STREAMS_PIPE;
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
 * @brief Gets caps from tensors info.
 */
GstCaps *
ml_tensors_info_get_caps (const ml_tensors_info_s * info)
{
  GstCaps *caps;
  GstTensorsConfig config;

  if (!info)
    return NULL;

  ml_tensors_info_copy_from_ml (&config.info, info);

  /* set framerate 0/1 */
  config.rate_n = 0;
  config.rate_d = 1;

  /* Supposed input type is single tensor if the number of tensors is 1. */
  if (config.info.num_tensors == 1) {
    GstTensorConfig c;

    gst_tensor_info_copy (&c.info, &config.info.info[0]);
    c.rate_n = 0;
    c.rate_d = 1;

    caps = gst_tensor_caps_from_config (&c);
    gst_tensor_info_free (&c.info);
  } else {
    caps = gst_tensors_caps_from_config (&config);
  }

  gst_tensors_info_free (&config.info);
  return caps;
}

/**
 * @brief Checks the availability of the given execution environments.
 */
int
ml_check_nnfw_availability (ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw,
    bool * available)
{
  check_feature_state ();

  if (!available)
    return ML_ERROR_INVALID_PARAMETER;

  /* init false */
  *available = false;

  switch (nnfw) {
    case ML_NNFW_TYPE_TENSORFLOW_LITE:
      if (nnstreamer_filter_find ("tensorflow-lite") == NULL) {
        ml_logw ("Tensorflow-lite is not supported.");
        goto done;
      }
      break;
    case ML_NNFW_TYPE_TENSORFLOW:
      if (nnstreamer_filter_find ("tensorflow") == NULL) {
        ml_logw ("Tensorflow is not supported.");
        goto done;
      }
      break;
    case ML_NNFW_TYPE_NNFW:
      {
        /** @todo Need to check method for NNFW */
        ml_logw ("NNFW is not supported.");
        goto done;
      }
      break;
    default:
      break;
  }

  *available = true;

done:
  return ML_ERROR_NONE;
}

/**
 * @brief Checks whether machine_learning.inference feature is enabled or not.
 */
int
ml_get_feature_enabled (void)
{
  ml_initialize_feature_status ();

#if defined(__TIZEN__)
  {
    int ret;
    int feature_enabled;

    g_mutex_lock (&feature_info->mutex);
    feature_enabled = feature_info->feature_status;
    g_mutex_unlock (&feature_info->mutex);

    if (0 == feature_enabled) {
      ml_loge ("machine_learning.inference NOT supported");
      return ML_ERROR_NOT_SUPPORTED;
    } else if (-1 == feature_enabled) {
      bool ml_inf_supported = false;
      ret = system_info_get_platform_bool(ML_INF_FEATURE_PATH, &ml_inf_supported);
      if (0 == ret) {
        if (false == ml_inf_supported) {
          ml_loge ("machine_learning.inference NOT supported");
          ml_set_feature_status (0);
          return ML_ERROR_NOT_SUPPORTED;
        }

        ml_set_feature_status (1);
      } else {
        switch (ret) {
          case SYSTEM_INFO_ERROR_INVALID_PARAMETER:
            ml_loge ("failed to get feature value because feature key is not vaild");
            ret = ML_ERROR_NOT_SUPPORTED;
            break;

          case SYSTEM_INFO_ERROR_IO_ERROR:
            ml_loge ("failed to get feature value because of input/output error");
            ret = ML_ERROR_NOT_SUPPORTED;
            break;

          case SYSTEM_INFO_ERROR_PERMISSION_DENIED:
            ml_loge ("failed to get feature value because of permission denied");
            ret = ML_ERROR_PERMISSION_DENIED;
            break;

          default:
            ml_loge ("failed to get feature value because of unknown error");
            ret = ML_ERROR_NOT_SUPPORTED;
            break;
        }
        return ret;
      }
    }
  }
#endif
  return ML_ERROR_NONE;
}

/**
 * @brief Set the feature status of machine_learning.inference.
 */
int
ml_set_feature_status (int status)
{
  ml_initialize_feature_status ();
  g_mutex_lock (&feature_info->mutex);

  /**
   * Update feature status
   * -1: Not checked yet, 0: Not supported, 1: Supported
   */
  feature_info->feature_status = status;

  g_mutex_unlock (&feature_info->mutex);
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

    restricted = nnsconf_get_custom_value_bool ("element-restriction", "enable_element_restriction", FALSE);
    if (restricted) {
      gchar *elements;

      /* check white-list of available plugins */
      elements = nnsconf_get_custom_value_string ("element-restriction", "restricted_elements");
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
