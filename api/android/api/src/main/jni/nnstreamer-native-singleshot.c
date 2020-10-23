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
 * @file	nnstreamer-native-singleshot.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/**
 * @brief Private data for SingleShot class.
 */
typedef struct
{
  ml_tensors_info_h out_info;
  jobject out_info_obj;
} singleshot_priv_data_s;

/**
 * @brief Release private data in pipeline info.
 */
static void
nns_singleshot_priv_free (gpointer data, JNIEnv * env)
{
  singleshot_priv_data_s *priv = (singleshot_priv_data_s *) data;

  ml_tensors_info_destroy (priv->out_info);
  if (priv->out_info_obj)
    (*env)->DeleteGlobalRef (env, priv->out_info_obj);

  g_free (priv);
}

/**
 * @brief Update output info in private data.
 */
static gboolean
nns_singleshot_priv_set_info (pipeline_info_s * pipe_info, JNIEnv * env)
{
  ml_single_h single;
  singleshot_priv_data_s *priv;
  ml_tensors_info_h out_info;
  jobject obj_info = NULL;

  single = pipe_info->pipeline_handle;
  priv = (singleshot_priv_data_s *) pipe_info->priv_data;

  if (ml_single_get_output_info (single, &out_info) != ML_ERROR_NONE) {
    nns_loge ("Failed to get output info.");
    return FALSE;
  }

  if (!ml_tensors_info_is_equal (out_info, priv->out_info)) {
    if (!nns_convert_tensors_info (pipe_info, env, out_info, &obj_info)) {
      nns_loge ("Failed to convert output info.");
      ml_tensors_info_destroy (out_info);
      return FALSE;
    }

    ml_tensors_info_destroy (priv->out_info);
    priv->out_info = out_info;

    if (priv->out_info_obj)
      (*env)->DeleteGlobalRef (env, priv->out_info_obj);
    priv->out_info_obj = (*env)->NewGlobalRef (env, obj_info);
    (*env)->DeleteLocalRef (env, obj_info);
  }

  return TRUE;
}

/**
 * @brief Native method for single-shot API.
 */
jlong
nns_native_single_open (JNIEnv * env, jobject thiz,
    jobjectArray models, jobject in, jobject out, jint fw_type, jstring option)
{
  pipeline_info_s *pipe_info = NULL;
  singleshot_priv_data_s *priv;
  ml_single_h single = NULL;
  ml_single_preset info = { 0, };
  gboolean opened = FALSE;

  pipe_info = nns_construct_pipe_info (env, thiz, NULL, NNS_PIPE_TYPE_SINGLE);
  if (pipe_info == NULL) {
    nns_loge ("Failed to create pipe info.");
    goto done;
  }

  /* parse in/out tensors information */
  if (in) {
    if (!nns_parse_tensors_info (pipe_info, env, in, &info.input_info)) {
      nns_loge ("Failed to parse input tensor.");
      goto done;
    }
  }

  if (out) {
    if (!nns_parse_tensors_info (pipe_info, env, out, &info.output_info)) {
      nns_loge ("Failed to parse output tensor.");
      goto done;
    }
  }

  /* nnfw type and hw resource */
  if (!nns_get_nnfw_type (fw_type, &info.nnfw)) {
    nns_loge ("Failed, unsupported framework (%d).", fw_type);
    goto done;
  }

  info.hw = ML_NNFW_HW_ANY;

  /* parse models */
  if (models) {
    GString *model_str;
    jsize i, models_count;

    model_str = g_string_new (NULL);
    models_count = (*env)->GetArrayLength (env, models);

    for (i = 0; i < models_count; i++) {
      jstring model = (jstring) (*env)->GetObjectArrayElement (env, models, i);
      const char *model_path = (*env)->GetStringUTFChars (env, model, NULL);

      g_string_append (model_str, model_path);
      if (i < models_count - 1) {
        g_string_append (model_str, ",");
      }

      (*env)->ReleaseStringUTFChars (env, model, model_path);
      (*env)->DeleteLocalRef (env, model);
    }

    info.models = g_string_free (model_str, FALSE);
  } else {
    nns_loge ("Failed to get model file.");
    goto done;
  }

  /* parse option string */
  if (option) {
    const char *option_str = (*env)->GetStringUTFChars (env, option, NULL);

    info.custom_option = g_strdup (option_str);
    (*env)->ReleaseStringUTFChars (env, option, option_str);
  }

  if (ml_single_open_custom (&single, &info) != ML_ERROR_NONE) {
    nns_loge ("Failed to create the pipeline.");
    goto done;
  }

  pipe_info->pipeline_handle = single;

  /* set private date */
  priv = g_new0 (singleshot_priv_data_s, 1);
  ml_tensors_info_create (&priv->out_info);
  nns_set_priv_data (pipe_info, priv, nns_singleshot_priv_free);

  if (!nns_singleshot_priv_set_info (pipe_info, env)) {
    nns_loge ("Failed to set the metadata.");
    goto done;
  }

  opened = TRUE;

done:
  ml_tensors_info_destroy (info.input_info);
  ml_tensors_info_destroy (info.output_info);
  g_free (info.models);
  g_free (info.custom_option);

  if (!opened) {
    nns_destroy_pipe_info (pipe_info, env);
    pipe_info = NULL;
  }

  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for single-shot API.
 */
void
nns_native_single_close (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for single-shot API.
 */
jobject
nns_native_single_invoke (JNIEnv * env, jobject thiz, jlong handle, jobject in)
{
  pipeline_info_s *pipe_info;
  singleshot_priv_data_s *priv;
  ml_single_h single;
  ml_tensors_data_h in_data, out_data;
  jobject result = NULL;
  gboolean failed = FALSE;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  priv = (singleshot_priv_data_s *) pipe_info->priv_data;
  single = pipe_info->pipeline_handle;
  in_data = out_data = NULL;

  if (!nns_parse_tensors_data (pipe_info, env, in, FALSE, &in_data, NULL)) {
    nns_loge ("Failed to parse input tensors data.");
    failed = TRUE;
    goto done;
  }

  /* create output object and get the direct buffer address */
  if (!nns_create_tensors_data_object (pipe_info, env, priv->out_info_obj, &result) ||
      !nns_parse_tensors_data (pipe_info, env, result, FALSE, &out_data, NULL)) {
    nns_loge ("Failed to create output tensors object.");
    failed = TRUE;
    goto done;
  }

  if (ml_single_invoke_fast (single, in_data, out_data) != ML_ERROR_NONE) {
    nns_loge ("Failed to invoke the model.");
    failed = TRUE;
    goto done;
  }

done:
  if (failed) {
    if (result) {
      (*env)->DeleteLocalRef (env, result);
      result = NULL;
    }
  }

  /* do not free input/output tensors (direct access from object) */
  g_free (in_data);
  g_free (out_data);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jobject
nns_native_single_get_input_info (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h info;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  if (ml_single_get_input_info (single, &info) != ML_ERROR_NONE) {
    nns_loge ("Failed to get input info.");
    goto done;
  }

  if (!nns_convert_tensors_info (pipe_info, env, info, &result)) {
    nns_loge ("Failed to convert input info.");
    result = NULL;
  }

done:
  ml_tensors_info_destroy (info);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jobject
nns_native_single_get_output_info (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h info;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  if (ml_single_get_output_info (single, &info) != ML_ERROR_NONE) {
    nns_loge ("Failed to get output info.");
    goto done;
  }

  if (!nns_convert_tensors_info (pipe_info, env, info, &result)) {
    nns_loge ("Failed to convert output info.");
    result = NULL;
  }

done:
  ml_tensors_info_destroy (info);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jboolean
nns_native_single_set_prop (JNIEnv * env, jobject thiz, jlong handle,
    jstring name, jstring value)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  jboolean ret = JNI_FALSE;

  const char *prop_name = (*env)->GetStringUTFChars (env, name, NULL);
  const char *prop_value = (*env)->GetStringUTFChars (env, value, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  /* update info when changing the tensors information */
  if (ml_single_set_property (single, prop_name, prop_value) == ML_ERROR_NONE &&
      nns_singleshot_priv_set_info (pipe_info, env)) {
    ret = JNI_TRUE;
  } else {
    nns_loge ("Failed to set the property (%s:%s).", prop_name, prop_value);
  }

  (*env)->ReleaseStringUTFChars (env, name, prop_name);
  (*env)->ReleaseStringUTFChars (env, name, prop_value);
  return ret;
}

/**
 * @brief Native method for single-shot API.
 */
jstring
nns_native_single_get_prop (JNIEnv * env, jobject thiz, jlong handle,
    jstring name)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;

  const char *prop_name = (*env)->GetStringUTFChars (env, name, NULL);
  char *prop_value = NULL;
  jstring value = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  if (ml_single_get_property (single, prop_name, &prop_value) == ML_ERROR_NONE) {
    if (!prop_value) {
      /* null string means error in java, return empty string. */
      prop_value = g_strdup ("");
    }

    value = (*env)->NewStringUTF (env, prop_value);
    g_free (prop_value);
  } else {
    nns_loge ("Failed to get the property (%s).", prop_name);
  }

  (*env)->ReleaseStringUTFChars (env, name, prop_name);
  return value;
}

/**
 * @brief Native method for single-shot API.
 */
jboolean
nns_native_single_set_timeout (JNIEnv * env, jobject thiz, jlong handle,
    jint timeout)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  if (ml_single_set_timeout (single, (unsigned int) timeout) != ML_ERROR_NONE) {
    nns_loge ("Failed to set the timeout.");
    return JNI_FALSE;
  }

  nns_logi ("Successfully set the timeout, %d milliseconds.", timeout);
  return JNI_TRUE;
}

/**
 * @brief Native method for single-shot API.
 */
jboolean
nns_native_single_set_input_info (JNIEnv * env, jobject thiz, jlong handle,
    jobject in)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h in_info = NULL;
  jboolean ret = JNI_FALSE;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  if (!nns_parse_tensors_info (pipe_info, env, in, &in_info)) {
    nns_loge ("Failed to parse input tensor.");
    goto done;
  }

  if (ml_single_set_input_info (single, in_info) == ML_ERROR_NONE &&
      nns_singleshot_priv_set_info (pipe_info, env)) {
    ret = JNI_TRUE;
  } else {
    nns_loge ("Failed to set input info.");
  }

done:
  ml_tensors_info_destroy (in_info);
  return ret;
}
