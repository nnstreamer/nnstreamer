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
  jmethodID mid_info;
  ml_tensors_info_h in_info;
  jobject in_info_obj;
} customfilter_priv_data_s;

/**
 * @brief Table to handle custom-filter.
 */
static GHashTable *g_customfilters = NULL;

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

  if (priv->in_info && ml_tensors_info_is_equal (in_info, priv->in_info)) {
    /* do nothing, tensors info is equal. */
    return TRUE;
  }

  if (!nns_convert_tensors_info (pipe_info, env, in_info, &obj_info)) {
    nns_loge ("Failed to convert tensors info.");
    return FALSE;
  }

  if (priv->in_info_obj)
    (*env)->DeleteGlobalRef (env, priv->in_info_obj);

  if (priv->in_info)
    ml_tensors_info_free (priv->in_info);
  else
    ml_tensors_info_create (&priv->in_info);

  ml_tensors_info_clone (priv->in_info, in_info);
  priv->in_info_obj = (*env)->NewGlobalRef (env, obj_info);
  (*env)->DeleteLocalRef (env, obj_info);
  return TRUE;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework.
 * @param prop The property of tensor_filter instance
 * @param private_data Sub-plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. Non-zero if error.
 */
static int
nns_customfilter_invoke (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  pipeline_info_s *pipe_info = NULL;
  customfilter_priv_data_s *priv;
  ml_tensors_data_h in_data, out_data;
  ml_tensors_info_h in_info;
  ml_tensors_data_s *_data;
  JNIEnv *env;
  jobject obj_in_data, obj_out_data;
  guint i;
  int ret = -1;

  /* get pipe info and init */
  pipe_info = g_hash_table_lookup (g_customfilters, prop->fwname);
  g_return_val_if_fail (pipe_info, -1);

  env = nns_get_jni_env (pipe_info);
  g_return_val_if_fail (env, -1);

  in_data = out_data = NULL;
  in_info = NULL;
  obj_in_data = obj_out_data = NULL;
  priv = (customfilter_priv_data_s *) pipe_info->priv_data;

  if (ml_tensors_data_create_no_alloc (NULL, &in_data) != ML_ERROR_NONE) {
    nns_loge ("Failed to create handle for input tensors data.");
    goto done;
  }

  if (ml_tensors_info_create (&in_info) != ML_ERROR_NONE) {
    nns_loge ("Failed to create handle for input tensors info.");
    goto done;
  }

  /* convert to c-api data type */
  _data = (ml_tensors_data_s *) in_data;
  _data->num_tensors = prop->input_meta.num_tensors;
  for (i = 0; i < _data->num_tensors; i++) {
    _data->tensors[i].tensor = input[i].data;
    _data->tensors[i].size = input[i].size;
  }

  ml_tensors_info_copy_from_gst (in_info, &prop->input_meta);
  if (!nns_customfilter_priv_set_in_info (pipe_info, env, in_info)) {
    goto done;
  }

  /* convert to data object */
  if (!nns_convert_tensors_data (pipe_info, env, in_data, priv->in_info_obj,
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

  if (!nns_parse_tensors_data (pipe_info, env, obj_out_data, &out_data, NULL)) {
    nns_loge ("Failed to parse output data.");
    goto done;
  }

  /* set output data */
  _data = (ml_tensors_data_s *) out_data;
  for (i = 0; i < _data->num_tensors; i++) {
    output[i].data = _data->tensors[i].tensor;

    if (_data->tensors[i].size != output[i].size) {
      nns_logw ("The result has different buffer size at index %d [%zd:%zd]",
          i, output[i].size, _data->tensors[i].size);
      output[i].size = _data->tensors[i].size;
    }
  }

  /* callback finished */
  ret = 0;

done:
  if (obj_in_data)
    (*env)->DeleteLocalRef (env, obj_in_data);
  if (obj_out_data)
    (*env)->DeleteLocalRef (env, obj_out_data);

  g_free (in_data);
  g_free (out_data);
  ml_tensors_info_destroy (in_info);
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
nns_customfilter_set_dimension (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorsInfo * in_info,
    GstTensorsInfo * out_info)
{
  pipeline_info_s *pipe_info = NULL;
  customfilter_priv_data_s *priv;
  ml_tensors_info_h in, out;
  jobject obj_in_info, obj_out_info;
  JNIEnv *env;
  int ret = -1;

  /* get pipe info and init */
  pipe_info = g_hash_table_lookup (g_customfilters, prop->fwname);
  g_return_val_if_fail (pipe_info, -1);

  env = nns_get_jni_env (pipe_info);
  g_return_val_if_fail (env, -1);

  in = out = NULL;
  obj_in_info = obj_out_info = NULL;
  priv = (customfilter_priv_data_s *) pipe_info->priv_data;

  if (ml_tensors_info_create (&in) != ML_ERROR_NONE) {
    nns_loge ("Failed to create handle for input tensors info.");
    goto done;
  }

  /* convert to c-api data type */
  ml_tensors_info_copy_from_gst (in, in_info);

  if (!nns_convert_tensors_info (pipe_info, env, in, &obj_in_info)) {
    nns_loge ("Failed to convert input tensors info to data object.");
    goto done;
  }

  /* call output info callback */
  obj_out_info = (*env)->CallObjectMethod (env, pipe_info->instance,
      priv->mid_info, obj_in_info);

  if ((*env)->ExceptionCheck (env)) {
    nns_loge ("Failed to call the custom-info callback.");
    (*env)->ExceptionClear (env);
    goto done;
  }

  if (!nns_parse_tensors_info (pipe_info, env, obj_out_info, &out)) {
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

  ml_tensors_info_destroy (in);
  ml_tensors_info_destroy (out);
  return ret;
}

/**
 * @brief Native method for custom filter.
 */
jlong
Java_org_nnsuite_nnstreamer_CustomFilter_nativeInitialize (JNIEnv * env,
    jobject thiz, jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  customfilter_priv_data_s *priv;
  GstTensorFilterFramework *fw = NULL;
  const char *filter_name = (*env)->GetStringUTFChars (env, name, NULL);

  nns_logd ("Try to add custom-filter %s.", filter_name);

  if (nnstreamer_filter_find (filter_name)) {
    nns_logw ("Custom-filter %s already exists.", filter_name);
    goto done;
  }

  /* prepare filter-framework */
  fw = g_new0 (GstTensorFilterFramework, 1);
  if (fw == NULL) {
    nns_loge ("Failed to allocate memory for filter framework.");
    goto done;
  }

  fw->version = GST_TENSOR_FILTER_FRAMEWORK_V0;
  fw->name = g_strdup (filter_name);
  fw->allocate_in_invoke = TRUE;
  fw->run_without_model = TRUE;
  fw->invoke_NN = nns_customfilter_invoke;
  fw->setInputDimension = nns_customfilter_set_dimension;

  /* register custom-filter */
  if (!nnstreamer_filter_probe (fw)) {
    nns_loge ("Failed to register custom-filter %s.", filter_name);
    g_free (fw->name);
    g_free (fw);
    goto done;
  }

  pipe_info = nns_construct_pipe_info (env, thiz, fw, NNS_PIPE_TYPE_CUSTOM);
  if (pipe_info == NULL) {
    nns_loge ("Failed to create pipe info.");
    nnstreamer_filter_exit (fw->name);
    g_free (fw->name);
    g_free (fw);
    goto done;
  }

  /* add custom-filter handle to the table */
  g_mutex_lock (&pipe_info->lock);

  if (g_customfilters == NULL) {
    g_customfilters =
        g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
  }

  g_assert (g_hash_table_insert (g_customfilters, g_strdup (filter_name),
          pipe_info));

  g_mutex_unlock (&pipe_info->lock);

  priv = g_new0 (customfilter_priv_data_s, 1);
  priv->mid_invoke = (*env)->GetMethodID (env, pipe_info->cls, "invoke",
      "(L" NNS_CLS_TDATA ";)L" NNS_CLS_TDATA ";");
  priv->mid_info = (*env)->GetMethodID (env, pipe_info->cls, "getOutputInfo",
      "(L" NNS_CLS_TINFO ";)L" NNS_CLS_TINFO ";");

  nns_set_priv_data (pipe_info, priv, nns_customfilter_priv_free);

done:
  (*env)->ReleaseStringUTFChars (env, name, filter_name);
  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for custom filter.
 */
void
Java_org_nnsuite_nnstreamer_CustomFilter_nativeDestroy (JNIEnv * env,
    jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  GstTensorFilterFramework *fw = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
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
