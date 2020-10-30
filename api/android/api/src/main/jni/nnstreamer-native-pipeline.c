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
 * @file	nnstreamer-native-pipeline.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/**
 * @brief Private data for Pipeline class.
 */
typedef struct
{
  jmethodID mid_state_cb;
  jmethodID mid_sink_cb;
} pipeline_priv_data_s;

/**
 * @brief Private data for sink node.
 */
typedef struct
{
  ml_tensors_info_h out_info;
  jobject out_info_obj;
} pipeline_sink_priv_data_s;

/**
 * @brief Release private data in pipeline info.
 */
static void
nns_pipeline_priv_free (gpointer data, JNIEnv * env)
{
  pipeline_priv_data_s *priv = (pipeline_priv_data_s *) data;

  /* nothing to free */
  g_free (priv);
}

/**
 * @brief Release private data in sink node.
 */
static void
nns_pipeline_sink_priv_free (gpointer data, JNIEnv * env)
{
  pipeline_sink_priv_data_s *priv = (pipeline_sink_priv_data_s *) data;

  ml_tensors_info_destroy (priv->out_info);
  if (priv->out_info_obj)
    (*env)->DeleteGlobalRef (env, priv->out_info_obj);

  g_free (priv);
}

/**
 * @brief Update output info in sink node data.
 */
static gboolean
nns_pipeline_sink_priv_set_out_info (element_data_s * item, JNIEnv * env,
    const ml_tensors_info_h out_info)
{
  pipeline_sink_priv_data_s *priv;
  jobject obj_info = NULL;

  if ((priv = item->priv_data) == NULL) {
    priv = g_new0 (pipeline_sink_priv_data_s, 1);
    ml_tensors_info_create (&priv->out_info);

    item->priv_data = priv;
    item->priv_destroy_func = nns_pipeline_sink_priv_free;
  }

  if (ml_tensors_info_is_equal (out_info, priv->out_info)) {
    /* do nothing, tensors info is equal. */
    return TRUE;
  }

  if (!nns_convert_tensors_info (item->pipe_info, env, out_info, &obj_info)) {
    nns_loge ("Failed to convert output info.");
    return FALSE;
  }

  ml_tensors_info_free (priv->out_info);
  ml_tensors_info_clone (priv->out_info, out_info);

  if (priv->out_info_obj)
    (*env)->DeleteGlobalRef (env, priv->out_info_obj);
  priv->out_info_obj = (*env)->NewGlobalRef (env, obj_info);
  (*env)->DeleteLocalRef (env, obj_info);
  return TRUE;
}

/**
 * @brief Pipeline state change callback.
 */
static void
nns_pipeline_state_cb (ml_pipeline_state_e state, void *user_data)
{
  pipeline_info_s *pipe_info;
  pipeline_priv_data_s *priv;
  jint new_state = (jint) state;
  JNIEnv *env;

  pipe_info = (pipeline_info_s *) user_data;
  priv = (pipeline_priv_data_s *) pipe_info->priv_data;

  if ((env = nns_get_jni_env (pipe_info)) == NULL) {
    nns_logw ("Cannot get jni env in the state callback.");
    return;
  }

  (*env)->CallVoidMethod (env, pipe_info->instance, priv->mid_state_cb,
      new_state);

  if ((*env)->ExceptionCheck (env)) {
    nns_loge ("Failed to call the callback method.");
    (*env)->ExceptionClear (env);
  }
}

/**
 * @brief New data callback for sink node.
 */
static void
nns_sink_data_cb (const ml_tensors_data_h data, const ml_tensors_info_h info,
    void *user_data)
{
  element_data_s *item;
  pipeline_info_s *pipe_info;
  pipeline_priv_data_s *priv;
  pipeline_sink_priv_data_s *priv_sink;
  jobject obj_data = NULL;
  JNIEnv *env;

  item = (element_data_s *) user_data;
  pipe_info = item->pipe_info;

  if ((env = nns_get_jni_env (pipe_info)) == NULL) {
    nns_logw ("Cannot get jni env in the sink callback.");
    return;
  }

  /* cache output tensors info */
  if (!nns_pipeline_sink_priv_set_out_info (item, env, info)) {
    return;
  }

  priv = (pipeline_priv_data_s *) pipe_info->priv_data;
  priv_sink = (pipeline_sink_priv_data_s *) item->priv_data;

  if (nns_convert_tensors_data (pipe_info, env, data, priv_sink->out_info_obj,
          &obj_data)) {
    jstring sink_name = (*env)->NewStringUTF (env, item->name);

    (*env)->CallVoidMethod (env, pipe_info->instance, priv->mid_sink_cb,
        sink_name, obj_data);

    if ((*env)->ExceptionCheck (env)) {
      nns_loge ("Failed to call the callback method.");
      (*env)->ExceptionClear (env);
    }

    (*env)->DeleteLocalRef (env, sink_name);
    (*env)->DeleteLocalRef (env, obj_data);
  } else {
    nns_loge ("Failed to convert the result to data object.");
  }
}

/**
 * @brief Get sink handle.
 */
static void *
nns_get_sink_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_SINK;
  ml_pipeline_sink_h handle;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_sink_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get sink handle and register to table */
    element_data_s *item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      nns_loge ("Failed to allocate memory for sink handle data.");
      return NULL;
    }

    status = ml_pipeline_sink_register (pipe, element_name, nns_sink_data_cb,
        item, &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get sink node %s.", element_name);
      g_free (item);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_handle (pipe_info, element_name, item)) {
      nns_loge ("Failed to add sink node %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Get src handle.
 */
static void *
nns_get_src_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_SRC;
  ml_pipeline_src_h handle;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_src_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get src handle and register to table */
    status = ml_pipeline_src_get_handle (pipe, element_name, &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get src node %s.", element_name);
      return NULL;
    }

    element_data_s *item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      nns_loge ("Failed to allocate memory for src handle data.");
      ml_pipeline_src_release_handle (handle);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_handle (pipe_info, element_name, item)) {
      nns_loge ("Failed to add src node %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Get switch handle.
 */
static void *
nns_get_switch_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_SWITCH;
  ml_pipeline_switch_h handle;
  ml_pipeline_switch_e switch_type;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_switch_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get switch handle and register to table */
    status = ml_pipeline_switch_get_handle (pipe, element_name, &switch_type,
        &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get switch %s.", element_name);
      return NULL;
    }

    element_data_s *item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      nns_loge ("Failed to allocate memory for switch handle data.");
      ml_pipeline_switch_release_handle (handle);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_handle (pipe_info, element_name, item)) {
      nns_loge ("Failed to add switch %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Get valve handle.
 */
static void *
nns_get_valve_handle (pipeline_info_s * pipe_info, const gchar * element_name)
{
  const nns_element_type_e etype = NNS_ELEMENT_TYPE_VALVE;
  ml_pipeline_valve_h handle;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_valve_h) nns_get_element_handle (pipe_info,
      element_name, etype);
  if (handle == NULL) {
    /* get valve handle and register to table */
    status = ml_pipeline_valve_get_handle (pipe, element_name, &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get valve %s.", element_name);
      return NULL;
    }

    element_data_s *item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      nns_loge ("Failed to allocate memory for valve handle data.");
      ml_pipeline_valve_release_handle (handle);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = etype;
    item->handle = handle;
    item->pipe_info = pipe_info;

    if (!nns_add_element_handle (pipe_info, element_name, item)) {
      nns_loge ("Failed to add valve %s.", element_name);
      nns_free_element_data (item);
      return NULL;
    }
  }

  return handle;
}

/**
 * @brief Native method for pipeline API.
 */
jlong
nns_native_pipe_construct (JNIEnv * env, jobject thiz, jstring description,
    jboolean add_state_cb)
{
  pipeline_info_s *pipe_info = NULL;
  pipeline_priv_data_s *priv;
  ml_pipeline_h pipe;
  int status;
  const char *pipeline = (*env)->GetStringUTFChars (env, description, NULL);

  pipe_info = nns_construct_pipe_info (env, thiz, NULL, NNS_PIPE_TYPE_PIPELINE);
  if (pipe_info == NULL) {
    nns_loge ("Failed to create pipe info.");
    goto done;
  }

  priv = g_new0 (pipeline_priv_data_s, 1);
  priv->mid_state_cb =
      (*env)->GetMethodID (env, pipe_info->cls, "stateChanged", "(I)V");
  priv->mid_sink_cb =
      (*env)->GetMethodID (env, pipe_info->cls, "newDataReceived",
      "(Ljava/lang/String;L" NNS_CLS_TDATA ";)V");

  nns_set_priv_data (pipe_info, priv, nns_pipeline_priv_free);

  if (add_state_cb)
    status = ml_pipeline_construct (pipeline, nns_pipeline_state_cb, pipe_info,
        &pipe);
  else
    status = ml_pipeline_construct (pipeline, NULL, NULL, &pipe);

  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to create the pipeline.");
    nns_destroy_pipe_info (pipe_info, env);
    pipe_info = NULL;
  } else {
    pipe_info->pipeline_handle = pipe;
  }

done:
  (*env)->ReleaseStringUTFChars (env, description, pipeline);
  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for pipeline API.
 */
void
nns_native_pipe_destroy (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_start (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  int status;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  pipe = pipe_info->pipeline_handle;

  status = ml_pipeline_start (pipe);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to start the pipeline.");
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_stop (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  int status;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  pipe = pipe_info->pipeline_handle;

  status = ml_pipeline_stop (pipe);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to stop the pipeline.");
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

/**
 * @brief Native method for pipeline API.
 */
jint
nns_native_pipe_get_state (JNIEnv * env, jobject thiz, jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  ml_pipeline_state_e state;
  int status;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);
  pipe = pipe_info->pipeline_handle;

  status = ml_pipeline_get_state (pipe, &state);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to get the pipeline state.");
    state = ML_PIPELINE_STATE_UNKNOWN;
  }

  return (jint) state;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_input_data (JNIEnv * env, jobject thiz, jlong handle,
    jstring name, jobject in)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_src_h src;
  ml_tensors_data_h in_data = NULL;
  int status;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  src = (ml_pipeline_src_h) nns_get_src_handle (pipe_info, element_name);
  if (src == NULL) {
    goto done;
  }

  if (!nns_parse_tensors_data (pipe_info, env, in, FALSE, &in_data, NULL)) {
    nns_loge ("Failed to parse input data.");
    goto done;
  }

  status = ml_pipeline_src_input_data (src, in_data,
      ML_PIPELINE_BUF_POLICY_DO_NOT_FREE);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to input tensors data to source node %s.", element_name);
    goto done;
  }

  res = JNI_TRUE;

done:
  (*env)->ReleaseStringUTFChars (env, name, element_name);
  /* do not free input tensors (direct access from object) */
  g_free (in_data);
  return res;
}

/**
 * @brief Native method for pipeline API.
 */
jobjectArray
nns_native_pipe_get_switch_pads (JNIEnv * env, jobject thiz, jlong handle,
    jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_switch_h node;
  int status;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);
  char **pad_list = NULL;
  guint i, total;
  jobjectArray result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  node = (ml_pipeline_switch_h) nns_get_switch_handle (pipe_info, element_name);
  if (node == NULL) {
    goto done;
  }

  status = ml_pipeline_switch_get_pad_list (node, &pad_list);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to get the pad list of switch %s.", element_name);
    goto done;
  }

  total = g_strv_length (pad_list);

  /* set string array */
  if (total > 0) {
    jclass cls_string = (*env)->FindClass (env, "java/lang/String");

    result = (*env)->NewObjectArray (env, total, cls_string, NULL);
    if (result == NULL) {
      nns_loge ("Failed to allocate string array.");
      (*env)->DeleteLocalRef (env, cls_string);
      goto done;
    }

    for (i = 0; i < total; i++) {
      jstring pad = (*env)->NewStringUTF (env, pad_list[i]);

      (*env)->SetObjectArrayElement (env, result, i, pad);
      (*env)->DeleteLocalRef (env, pad);
    }

    (*env)->DeleteLocalRef (env, cls_string);
  }

done:
  g_strfreev (pad_list);
  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return result;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_select_switch_pad (JNIEnv * env, jobject thiz, jlong handle,
    jstring name, jstring pad)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_switch_h node;
  int status;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);
  const char *pad_name = (*env)->GetStringUTFChars (env, pad, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  node = (ml_pipeline_switch_h) nns_get_switch_handle (pipe_info, element_name);
  if (node == NULL) {
    goto done;
  }

  status = ml_pipeline_switch_select (node, pad_name);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to select switch pad %s.", pad_name);
    goto done;
  }

  res = JNI_TRUE;

done:
  (*env)->ReleaseStringUTFChars (env, name, element_name);
  (*env)->ReleaseStringUTFChars (env, pad, pad_name);
  return res;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_control_valve (JNIEnv * env, jobject thiz, jlong handle,
    jstring name, jboolean open)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_valve_h node;
  int status;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  node = (ml_pipeline_valve_h) nns_get_valve_handle (pipe_info, element_name);
  if (node == NULL) {
    goto done;
  }

  status = ml_pipeline_valve_set_open (node, (open == JNI_TRUE));
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to control valve %s.", element_name);
    goto done;
  }

  res = JNI_TRUE;

done:
  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return res;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_add_sink_cb (JNIEnv * env, jobject thiz, jlong handle,
    jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_sink_h sink;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  sink = (ml_pipeline_sink_h) nns_get_sink_handle (pipe_info, element_name);
  if (sink == NULL) {
    goto done;
  }

  res = JNI_TRUE;

done:
  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return res;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
nns_native_pipe_remove_sink_cb (JNIEnv * env, jobject thiz, jlong handle,
    jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_sink_h sink;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  /* get handle from table */
  sink = (ml_pipeline_sink_h) nns_get_element_handle (pipe_info, element_name,
      NNS_ELEMENT_TYPE_SINK);
  if (sink) {
    nns_remove_element_handle (pipe_info, element_name);
    res = JNI_TRUE;
  }

  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return res;
}
