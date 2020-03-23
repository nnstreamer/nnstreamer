/**
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
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
 * @brief Native method for single-shot API.
 */
jlong
Java_org_nnsuite_nnstreamer_SingleShot_nativeOpen (JNIEnv * env, jobject thiz,
    jobjectArray models, jobject in, jobject out, jint fw_type, jstring option)
{
  pipeline_info_s *pipe_info = NULL;
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
      jstring model_obj =
          (jstring) (*env)->GetObjectArrayElement (env, models, i);
      const char *model_path = (*env)->GetStringUTFChars (env, model_obj, NULL);

      g_string_append (model_str, model_path);
      if (i < models_count - 1) {
        g_string_append (model_str, ",");
      }

      (*env)->ReleaseStringUTFChars (env, model_obj, model_path);
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

  opened = TRUE;
  pipe_info->pipeline_handle = single;

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
Java_org_nnsuite_nnstreamer_SingleShot_nativeClose (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_org_nnsuite_nnstreamer_SingleShot_nativeInvoke (JNIEnv * env,
    jobject thiz, jlong handle, jobject in)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h cur_info, in_info, out_info;
  ml_tensors_data_h in_data, out_data;
  int status;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;
  cur_info = in_info = out_info = NULL;
  in_data = out_data = NULL;

  if (ml_single_get_input_info (single, &cur_info) != ML_ERROR_NONE) {
    nns_loge ("Failed to get input tensors info.");
    goto done;
  }

  if (!nns_parse_tensors_data (pipe_info, env, in, &in_data, &in_info)) {
    nns_loge ("Failed to parse input tensors data.");
    goto done;
  }

  if (in_info == NULL || ml_tensors_info_is_equal (cur_info, in_info)) {
    /* input tensors info is not changed */
    if (ml_single_get_output_info (single, &out_info) != ML_ERROR_NONE) {
      nns_loge ("Failed to get output tensors info.");
      goto done;
    }

    status = ml_single_invoke (single, in_data, &out_data);
  } else {
    /* input tensors info changed, call dynamic */
    status =
        ml_single_invoke_dynamic (single, in_data, in_info, &out_data,
        &out_info);
  }

  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to get the result from pipeline.");
    goto done;
  }

  if (!nns_convert_tensors_data (pipe_info, env, out_data, out_info, &result)) {
    nns_loge ("Failed to convert the result to data.");
    result = NULL;
  }

done:
  ml_tensors_data_destroy (in_data);
  ml_tensors_data_destroy (out_data);
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);
  ml_tensors_info_destroy (cur_info);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_org_nnsuite_nnstreamer_SingleShot_nativeGetInputInfo (JNIEnv * env,
    jobject thiz, jlong handle)
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
Java_org_nnsuite_nnstreamer_SingleShot_nativeGetOutputInfo (JNIEnv * env,
    jobject thiz, jlong handle)
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
Java_org_nnsuite_nnstreamer_SingleShot_nativeSetProperty (JNIEnv * env,
    jobject thiz, jlong handle, jstring name, jstring value)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  jboolean ret = JNI_FALSE;

  const char *prop_name = (*env)->GetStringUTFChars (env, name, NULL);
  const char *prop_value = (*env)->GetStringUTFChars (env, value, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  single = pipe_info->pipeline_handle;

  if (ml_single_set_property (single, prop_name, prop_value) == ML_ERROR_NONE) {
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
Java_org_nnsuite_nnstreamer_SingleShot_nativeGetProperty (JNIEnv * env,
    jobject thiz, jlong handle, jstring name)
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
Java_org_nnsuite_nnstreamer_SingleShot_nativeSetTimeout (JNIEnv * env,
    jobject thiz, jlong handle, jint timeout)
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
Java_org_nnsuite_nnstreamer_SingleShot_nativeSetInputInfo (JNIEnv * env,
    jobject thiz, jlong handle, jobject in)
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

  if (ml_single_set_input_info (single, in_info) != ML_ERROR_NONE) {
    nns_loge ("Failed to set input info.");
    goto done;
  }

  ret = JNI_TRUE;

done:
  ml_tensors_info_destroy (in_info);
  return ret;
}
