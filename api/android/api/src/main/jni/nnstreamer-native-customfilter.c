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
 * @file	nnstreamer-native-customfilter.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/**
 * @brief Table to handle custom-filter.
 */
static GHashTable *g_customfilters = NULL;

/**
 * @brief The mandatory callback for GstTensorFilterFramework.
 * @param prop The property of tensor_filter instance
 * @param private_data Sub-plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. Non-zero if error.
 */
static int
nns_customfilter_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  pipeline_info_s *pipe_info = NULL;
  ml_tensors_data_s *in_data, *out_data;
  ml_tensors_info_s *in_info, *out_info;
  JNIEnv *env;
  guint i;
  int ret = -1;

  /* get pipe info and init */
  pipe_info = g_hash_table_lookup (g_customfilters, prop->fwname);
  g_return_val_if_fail (pipe_info, -1);

  env = nns_get_jni_env (pipe_info);
  g_return_val_if_fail (env, -1);

  in_data = g_new0 (ml_tensors_data_s, 1);
  g_assert (in_data);

  out_data = g_new0 (ml_tensors_data_s, 1);
  g_assert (out_data);

  in_info = g_new0 (ml_tensors_info_s, 1);
  g_assert (in_info);

  out_info = g_new0 (ml_tensors_info_s, 1);
  g_assert (out_info);

  /* convert to c-api data type */
  in_data->num_tensors = prop->input_meta.num_tensors;
  for (i = 0; i < in_data->num_tensors; i++) {
    in_data->tensors[i].tensor = input[i].data;
    in_data->tensors[i].size = input[i].size;
  }

  ml_tensors_info_copy_from_gst (in_info, &prop->input_meta);
  ml_tensors_info_copy_from_gst (out_info, &prop->output_meta);

  /* call invoke callback */
  jobject obj_in_data, obj_out_data;
  jobject obj_in_info, obj_out_info;

  obj_in_data = obj_out_data = NULL;
  obj_in_info = obj_out_info = NULL;

  if (!nns_convert_tensors_info (pipe_info, env, in_info, &obj_in_info)) {
    nns_loge ("Failed to convert input info to info-object.");
    goto done;
  }

  if (!nns_convert_tensors_info (pipe_info, env, out_info, &obj_out_info)) {
    nns_loge ("Failed to convert output info to info-object.");
    goto done;
  }

  if (!nns_convert_tensors_data (pipe_info, env, in_data, &obj_in_data)) {
    nns_loge ("Failed to convert input data to data-object.");
    goto done;
  }

  jclass cls_custom = (*env)->GetObjectClass (env, pipe_info->instance);
  jmethodID mid_invoke = (*env)->GetMethodID (env, cls_custom, "invoke",
      "(Lorg/nnsuite/nnstreamer/TensorsData;"
      "Lorg/nnsuite/nnstreamer/TensorsInfo;"
      "Lorg/nnsuite/nnstreamer/TensorsInfo;)"
      "Lorg/nnsuite/nnstreamer/TensorsData;");

  obj_out_data = (*env)->CallObjectMethod (env, pipe_info->instance, mid_invoke,
      obj_in_data, obj_in_info, obj_out_info);
  if (!nns_parse_tensors_data (pipe_info, env, obj_out_data, out_data)) {
    nns_loge ("Failed to parse output data.");
    goto done;
  }

  /* set output data */
  for (i = 0; i < out_data->num_tensors; i++) {
    output[i].data = out_data->tensors[i].tensor;

    if (out_data->tensors[i].size != output[i].size) {
      nns_logw ("The result has different buffer size at index %d [%zd:%zd]",
          i, output[i].size, out_data->tensors[i].size);
      output[i].size = out_data->tensors[i].size;
    }
  }

  /* callback finished */
  ret = 0;

done:
  if (obj_in_data)
    (*env)->DeleteLocalRef (env, obj_in_data);
  if (obj_out_data)
    (*env)->DeleteLocalRef (env, obj_out_data);
  if (obj_in_info)
    (*env)->DeleteLocalRef (env, obj_in_info);
  if (obj_out_info)
    (*env)->DeleteLocalRef (env, obj_out_info);
  (*env)->DeleteLocalRef (env, cls_custom);

  g_free (in_data);
  g_free (out_data);
  ml_tensors_info_destroy ((ml_tensors_info_h) in_info);
  ml_tensors_info_destroy ((ml_tensors_info_h) out_info);
  return ret;
}

/**
 * @brief The optional callback for GstTensorFilterFramework.
 * @param prop The property of tensor_filter instance
 * @param private_data Sub-plugin's private data
 * @param[in] in_info The dimension and type of input tensors
 * @param[out] out_info The dimension and type of output tensors
 * @return 0 if OK. Non-zero if error.
 */
static int
nns_customfilter_set_dimension (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  pipeline_info_s *pipe_info = NULL;
  ml_tensors_info_s *in, *out;
  JNIEnv *env;
  int ret = -1;

  /* get pipe info and init */
  pipe_info = g_hash_table_lookup (g_customfilters, prop->fwname);
  g_return_val_if_fail (pipe_info, -1);

  env = nns_get_jni_env (pipe_info);
  g_return_val_if_fail (env, -1);

  in = g_new0 (ml_tensors_info_s, 1);
  g_assert (in);

  out = g_new0 (ml_tensors_info_s, 1);
  g_assert (out);

  /* convert to c-api data type */
  ml_tensors_info_copy_from_gst (in, in_info);

  /* call output info callback */
  jobject obj_in_info, obj_out_info;

  obj_in_info = obj_out_info = NULL;
  if (!nns_convert_tensors_info (pipe_info, env, in, &obj_in_info)) {
    nns_loge ("Failed to convert input tensors info to data object.");
    goto done;
  }

  jclass cls_custom = (*env)->GetObjectClass (env, pipe_info->instance);
  jmethodID mid_info = (*env)->GetMethodID (env, cls_custom, "getOutputInfo",
      "(Lorg/nnsuite/nnstreamer/TensorsInfo;)"
      "Lorg/nnsuite/nnstreamer/TensorsInfo;");

  obj_out_info = (*env)->CallObjectMethod (env, pipe_info->instance, mid_info, obj_in_info);
  if (!obj_out_info || !nns_parse_tensors_info (pipe_info, env, obj_out_info, out)) {
    nns_loge ("Failed to parse output info.");
    goto done;
  }

  /* set output data */
  ml_tensors_info_copy_from_ml (out_info, out);

  /* callback finished */
  ret = 0;

done:
  if (obj_in_info)
    (*env)->DeleteLocalRef (env, obj_in_info);
  if (obj_out_info)
    (*env)->DeleteLocalRef (env, obj_out_info);
  (*env)->DeleteLocalRef (env, cls_custom);

  ml_tensors_info_destroy ((ml_tensors_info_h) in);
  ml_tensors_info_destroy ((ml_tensors_info_h) out);
  return ret;
}

/**
 * @brief Native method for custom filter.
 */
jlong
Java_org_nnsuite_nnstreamer_CustomFilter_nativeInitialize (JNIEnv * env, jobject thiz,
    jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  GstTensorFilterFramework *fw = NULL;
  const char *filter_name = (*env)->GetStringUTFChars (env, name, NULL);

  nns_logd ("Try to add custom-filter %s.", filter_name);

  if (nnstreamer_filter_find (filter_name)) {
    nns_logw ("Custom-filter %s already exists.", filter_name);
    goto done;
  }

  /* prepare filter-framework */
  fw = g_new0 (GstTensorFilterFramework, 1);
  if (!fw) {
    nns_loge ("Failed to allocate memory for filter framework.");
    goto done;
  }

  fw->name = g_strdup (filter_name);
  fw->allocate_in_invoke = TRUE;
  fw->run_without_model = TRUE;
  fw->invoke_NN = nns_customfilter_invoke;
  fw->setInputDimension = nns_customfilter_set_dimension;

  if (!nnstreamer_filter_probe (fw)) {
    nns_loge ("Failed to register custom-filter %s.", filter_name);
    g_free (fw->name);
    g_free (fw);
    goto done;
  }

  pipe_info = nns_construct_pipe_info (env, thiz, fw, NNS_PIPE_TYPE_CUSTOM);

  /* add custom-filter handle to the table */
  g_mutex_lock (&pipe_info->lock);

  if (g_customfilters == NULL) {
    g_customfilters = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  }

  g_assert (g_hash_table_insert (g_customfilters, g_strdup (filter_name), pipe_info));

  g_mutex_unlock (&pipe_info->lock);

done:
  (*env)->ReleaseStringUTFChars (env, name, filter_name);
  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for custom filter.
 */
void
Java_org_nnsuite_nnstreamer_CustomFilter_nativeDestroy (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  GstTensorFilterFramework *fw = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  g_return_if_fail (pipe_info);

  fw = (GstTensorFilterFramework *) pipe_info->pipeline_handle;
  nns_logd ("Start to unregister custom-filter %s.", fw->name);

  g_mutex_lock (&pipe_info->lock);
  if (!g_hash_table_remove (g_customfilters, fw->name)) {
    nns_logw ("Failed to remove custom-filter %s.", fw->name);
  }
  g_mutex_unlock (&pipe_info->lock);

  nnstreamer_filter_exit (fw->name);
  g_free (fw->name);
  g_free (fw);

  nns_destroy_pipe_info (pipe_info, env);
}
