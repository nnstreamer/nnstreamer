/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file nnstreamer-capi-tizen-feature-check.c
 * @date 21 July 2020
 * @brief NNStreamer/C-API Tizen dependent functions.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#if !defined (__TIZEN__) || !defined (__FEATURE_CHECK_SUPPORT__)
#error "This file can be included only in Tizen."
#endif

#include <glib.h>
#include <system_info.h>

#include "nnstreamer-capi-private.h"

/**
 * @brief Tizen ML feature.
 */
#define ML_INF_FEATURE_PATH "tizen.org/feature/machine_learning.inference"

/**
 * @brief Internal struct to control tizen feature support (machine_learning.inference).
 * -1: Not checked yet, 0: Not supported, 1: Supported
 */
typedef struct
{
  GMutex mutex;
  feature_state_t feature_state;
} feature_info_s;

static feature_info_s *feature_info = NULL;

/**
 * @brief Internal function to initialize feature state.
 */
static void
ml_tizen_initialize_feature_state (void)
{
  if (feature_info == NULL) {
    feature_info = g_new0 (feature_info_s, 1);
    g_assert (feature_info);

    g_mutex_init (&feature_info->mutex);
    feature_info->feature_state = NOT_CHECKED_YET;
  }
}

/**
 * @brief Set the feature status of machine_learning.inference.
 */
int
ml_tizen_set_feature_state (int state)
{
  ml_tizen_initialize_feature_state ();
  g_mutex_lock (&feature_info->mutex);

  /**
   * Update feature status
   * -1: Not checked yet, 0: Not supported, 1: Supported
   */
  feature_info->feature_state = state;

  g_mutex_unlock (&feature_info->mutex);
  return ML_ERROR_NONE;
}

/**
 * @brief Checks whether machine_learning.inference feature is enabled or not.
 */
int
ml_tizen_get_feature_enabled (void)
{
  int ret;
  int feature_enabled;

  ml_tizen_initialize_feature_state ();

  g_mutex_lock (&feature_info->mutex);
  feature_enabled = feature_info->feature_state;
  g_mutex_unlock (&feature_info->mutex);

  if (NOT_SUPPORTED == feature_enabled) {
    ml_loge ("machine_learning.inference NOT supported");
    return ML_ERROR_NOT_SUPPORTED;
  } else if (NOT_CHECKED_YET == feature_enabled) {
    bool ml_inf_supported = false;
    ret =
        system_info_get_platform_bool (ML_INF_FEATURE_PATH, &ml_inf_supported);
    if (0 == ret) {
      if (false == ml_inf_supported) {
        ml_loge ("machine_learning.inference NOT supported");
        ml_tizen_set_feature_state (NOT_SUPPORTED);
        return ML_ERROR_NOT_SUPPORTED;
      }

      ml_tizen_set_feature_state (SUPPORTED);
    } else {
      switch (ret) {
        case SYSTEM_INFO_ERROR_INVALID_PARAMETER:
          ml_loge
              ("failed to get feature value because feature key is not vaild");
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

  return ML_ERROR_NONE;
}
