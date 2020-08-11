/**
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Library General Public License for more details.
 */

/**
 * @file	nnstreamer-native-customfilter.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/**
 * @brief Private data for CustomFilter class.
 */
typedef struct
{
  jmethodID mid_invoke;
  ml_tensors_info_h in_info;
  jobject in_info_obj;
} customfilter_priv_data_s;

/**
 * @brief Release private data in custom filter.
 */
static void
nns_customfilter_priv_free (gpointer data, JNIEnv * env)
{
  customfilter_priv_data_s *priv = (customfilter_priv_data_s *) data;

  ml_tensors_info_destroy (priv->in_info);
  if (priv->in_info_obj)
    (*env)->DeleteGlobalRef (env, priv->in_info_obj);

  g_free (priv);
}

/**
 * @brief Update input info in private data.
 */
static gboolean
nns_customfilter_priv_set_in_info (pipeline_info_s * pipe_info, JNIEnv * env,
    ml_tensors_info_h in_info)
{
  customfilter_priv_data_s *priv;
  jobject obj_info = NULL;

  priv = (customfilter_priv_data_s *) pipe_info->priv_data;

  if (ml_tensors_info_is_equal (in_info, priv->in_info)) {
    /* do nothing, tensors info is equal. */
    return TRUE;
  }

  if (!nns_convert_tensors_info (pipe_info, env, in_info, &obj_info)) {
    nns_loge ("Failed to convert tensors info.");
    return FALSE;
  }

  ml_tensors_info_free (priv->in_info);
  ml_tensors_info_clone (priv->in_info, in_info);

  if (priv->in_info_obj)
    (*env)->DeleteGlobalRef (env, priv->in_info_obj);
  priv->in_info_obj = (*env)->NewGlobalRef (env, obj_info);
  (*env)->DeleteLocalRef (env, obj_info);
  return TRUE;
}

/**
 * @brief The mandatory callback for custom-filter execution.
 * @return 0 if OK. 1 to drop input buffer. Negative value if error.
 */
static int
nns_customfilter_invoke (const ml_tensors_data_h in, ml_tensors_data_h out,
    void *user_data)
{
  pipeline_info_s *pipe_info = NULL;
  customfilter_priv_data_s *priv;
  JNIEnv *env;
  jobject obj_in_data, obj_out_data;
  int ret = -1;

  /* get pipe info and init */
  pipe_info = (pipeline_info_s *) user_data;
  g_return_val_if_fail (pipe_info, -1);

  env = nns_get_jni_env (pipe_info);
  g_return_val_if_fail (env, -1);

  obj_in_data = obj_out_data = NULL;
  priv = (customfilter_priv_data_s *) pipe_info->priv_data;

  /* convert to data object */
  if (!nns_convert_tensors_data (pipe_info, env, in, priv->in_info_obj,
          &obj_in_data)) {
    nns_loge ("Failed to convert input data to data-object.");
    goto done;
  }

  /* call invoke callback */
  obj_out_data = (*env)->CallObjectMethod (env, pipe_info->instance,
      priv->mid_invoke, obj_in_data);

  if ((*env)->ExceptionCheck (env)) {
    nns_loge ("Failed to call the custom-invoke callback.");
    (*env)->ExceptionClear (env);
    goto done;
  }

  if (obj_out_data == NULL) {
    /* drop current buffer */
    ret = 1;
    goto done;
  }

  if (!nns_parse_tensors_data (pipe_info, env, obj_out_data, &out, NULL)) {
    nns_loge ("Failed to parse output data.");
    goto done;
  }

  /* callback finished */
  ret = 0;

done:
  if (obj_in_data)
    (*env)->DeleteLocalRef (env, obj_in_data);
  if (obj_out_data)
    (*env)->DeleteLocalRef (env, obj_out_data);

  return ret;
}

/**
 * @brief Native method for custom filter.
 */
jlong
nns_native_custom_initialize (JNIEnv * env, jobject thiz, jstring name,
    jobject in, jobject out)
{
  pipeline_info_s *pipe_info = NULL;
  customfilter_priv_data_s *priv;
  ml_custom_easy_filter_h custom;
  ml_tensors_info_h in_info, out_info;
  gboolean is_done = FALSE;
  int status;
  const char *model_name = (*env)->GetStringUTFChars (env, name, NULL);

  nns_logd ("Try to add custom-filter %s.", model_name);
  in_info = out_info = NULL;

  pipe_info = nns_construct_pipe_info (env, thiz, NULL, NNS_PIPE_TYPE_CUSTOM);
  if (pipe_info == NULL) {
    nns_loge ("Failed to create pipe info.");
    goto done;
  }

  priv = g_new0 (customfilter_priv_data_s, 1);
  priv->mid_invoke = (*env)->GetMethodID (env, pipe_info->cls, "invoke",
      "(L" NNS_CLS_TDATA ";)L" NNS_CLS_TDATA ";");
  ml_tensors_info_create (&priv->in_info);

  nns_set_priv_data (pipe_info, priv, nns_customfilter_priv_free);

  if (!nns_parse_tensors_info (pipe_info, env, in, &in_info)) {
    nns_loge ("Failed to parse input info.");
    goto done;
  }

  if (!nns_parse_tensors_info (pipe_info, env, out, &out_info)) {
    nns_loge ("Failed to parse output info.");
    goto done;
  }

  /* update input info */
  if (!nns_customfilter_priv_set_in_info (pipe_info, env, in_info)) {
    goto done;
  }

  status = ml_pipeline_custom_easy_filter_register (model_name,
      in_info, out_info, nns_customfilter_invoke, pipe_info, &custom);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to register custom-filter %s.", model_name);
    goto done;
  }

  pipe_info->pipeline_handle = custom;
  is_done = TRUE;

done:
  (*env)->ReleaseStringUTFChars (env, name, model_name);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  if (!is_done) {
    nns_destroy_pipe_info (pipe_info, env);
    pipe_info = NULL;
  }

  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for custom filter.
 */
void
nns_native_custom_destroy (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  nns_destroy_pipe_info (pipe_info, env);
}
