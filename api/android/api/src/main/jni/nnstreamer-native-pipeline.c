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
 * @brief Pipeline state change callback.
 */
static void
nns_pipeline_state_cb (ml_pipeline_state_e state, void *user_data)
{
  pipeline_info_s *pipe_info;

  pipe_info = (pipeline_info_s *) user_data;

  JNIEnv *env = nns_get_jni_env (pipe_info);
  if (env == NULL) {
    nns_logw ("Cannot get jni env in the state callback.");
    return;
  }

  jclass cls_pipeline = (*env)->GetObjectClass (env, pipe_info->instance);
  jmethodID mid_callback =
      (*env)->GetMethodID (env, cls_pipeline, "stateChanged", "(I)V");
  jint new_state = (jint) state;

  (*env)->CallVoidMethod (env, pipe_info->instance, mid_callback, new_state);

  if ((*env)->ExceptionCheck (env)) {
    nns_loge ("Failed to call the callback method.");
    (*env)->ExceptionClear (env);
  }

  (*env)->DeleteLocalRef (env, cls_pipeline);
}

/**
 * @brief New data callback for sink node.
 */
static void
nns_sink_data_cb (const ml_tensors_data_h data, const ml_tensors_info_h info,
    void *user_data)
{
  element_data_s *cb_data;
  pipeline_info_s *pipe_info;
  jobject obj_data = NULL;
  JNIEnv *env;

  cb_data = (element_data_s *) user_data;
  pipe_info = cb_data->pipe_info;

  if ((env = nns_get_jni_env (pipe_info)) == NULL) {
    nns_logw ("Cannot get jni env in the sink callback.");
    return;
  }

  if (nns_convert_tensors_data (pipe_info, env, data, info, &obj_data)) {
    /* method for sink callback */
    jclass cls_pipeline = (*env)->GetObjectClass (env, pipe_info->instance);
    jmethodID mid_callback =
        (*env)->GetMethodID (env, cls_pipeline, "newDataReceived",
        "(Ljava/lang/String;Lorg/nnsuite/nnstreamer/TensorsData;)V");
    jstring sink_name = (*env)->NewStringUTF (env, cb_data->name);

    (*env)->CallVoidMethod (env, pipe_info->instance, mid_callback, sink_name,
        obj_data);

    if ((*env)->ExceptionCheck (env)) {
      nns_loge ("Failed to call the callback method.");
      (*env)->ExceptionClear (env);
    }

    (*env)->DeleteLocalRef (env, sink_name);
    (*env)->DeleteLocalRef (env, cls_pipeline);
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
  ml_pipeline_sink_h handle;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle =
      (ml_pipeline_sink_h) nns_get_element_handle (pipe_info, element_name);
  if (handle == NULL) {
    /* get sink handle and register to table */
    element_data_s *item = g_new0 (element_data_s, 1);
    if (item == NULL) {
      nns_loge ("Failed to allocate memory for sink handle data.");
      return NULL;
    }

    status =
        ml_pipeline_sink_register (pipe, element_name, nns_sink_data_cb, item,
        &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get sink node %s.", element_name);
      g_free (item);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = NNS_ELEMENT_TYPE_SINK;
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
  ml_pipeline_src_h handle;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle = (ml_pipeline_src_h) nns_get_element_handle (pipe_info, element_name);
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
    item->type = NNS_ELEMENT_TYPE_SRC;
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
  ml_pipeline_switch_h handle;
  ml_pipeline_switch_e switch_type;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle =
      (ml_pipeline_switch_h) nns_get_element_handle (pipe_info, element_name);
  if (handle == NULL) {
    /* get switch handle and register to table */
    status =
        ml_pipeline_switch_get_handle (pipe, element_name, &switch_type,
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
    if (switch_type == ML_PIPELINE_SWITCH_INPUT_SELECTOR)
      item->type = NNS_ELEMENT_TYPE_SWITCH_IN;
    else
      item->type = NNS_ELEMENT_TYPE_SWITCH_OUT;
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
  ml_pipeline_valve_h handle;
  ml_pipeline_h pipe;
  int status;

  g_assert (pipe_info);
  pipe = pipe_info->pipeline_handle;

  handle =
      (ml_pipeline_valve_h) nns_get_element_handle (pipe_info, element_name);
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
    item->type = NNS_ELEMENT_TYPE_VALVE;
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeConstruct (JNIEnv * env,
    jobject thiz, jstring description, jboolean add_state_cb)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  int status;
  const char *pipeline = (*env)->GetStringUTFChars (env, description, NULL);

  pipe_info = nns_construct_pipe_info (env, thiz, NULL, NNS_PIPE_TYPE_PIPELINE);
  if (pipe_info == NULL) {
    nns_loge ("Failed to create pipe info.");
    goto done;
  }

  if (add_state_cb)
    status =
        ml_pipeline_construct (pipeline, nns_pipeline_state_cb, pipe_info,
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeDestroy (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
Java_org_nnsuite_nnstreamer_Pipeline_nativeStart (JNIEnv * env, jobject thiz,
    jlong handle)
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeStop (JNIEnv * env, jobject thiz,
    jlong handle)
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeGetState (JNIEnv * env, jobject thiz,
    jlong handle)
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeInputData (JNIEnv * env,
    jobject thiz, jlong handle, jstring name, jobject in)
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

  if (!nns_parse_tensors_data (pipe_info, env, in, &in_data, NULL)) {
    nns_loge ("Failed to parse input data.");
    goto done;
  }

  status = ml_pipeline_src_input_data (src, in_data,
      ML_PIPELINE_BUF_POLICY_AUTO_FREE);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to input tensors data to source node %s.", element_name);
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
jobjectArray
Java_org_nnsuite_nnstreamer_Pipeline_nativeGetSwitchPads (JNIEnv * env,
    jobject thiz, jlong handle, jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_switch_h node;
  int status;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);
  char **pad_list = NULL;
  guint i, total = 0;
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

  /* set string array */
  if (pad_list) {
    jclass cls_string = (*env)->FindClass (env, "java/lang/String");

    while (pad_list[total] != NULL)
      total++;

    result =
        (*env)->NewObjectArray (env, total, cls_string,
        (*env)->NewStringUTF (env, ""));
    if (result == NULL) {
      nns_loge ("Failed to allocate string array.");
      (*env)->DeleteLocalRef (env, cls_string);
      goto done;
    }

    for (i = 0; i < total; i++) {
      (*env)->SetObjectArrayElement (env, result, i, (*env)->NewStringUTF (env,
              pad_list[i]));
      g_free (pad_list[i]);
    }

    g_free (pad_list);
    pad_list = NULL;

    (*env)->DeleteLocalRef (env, cls_string);
  }

done:
  /* free pad list */
  if (pad_list) {
    i = 0;
    while (pad_list[i] != NULL) {
      g_free (pad_list[i]);
    }
    g_free (pad_list);
  }

  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return result;
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
Java_org_nnsuite_nnstreamer_Pipeline_nativeSelectSwitchPad (JNIEnv * env,
    jobject thiz, jlong handle, jstring name, jstring pad)
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeControlValve (JNIEnv * env,
    jobject thiz, jlong handle, jstring name, jboolean open)
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeAddSinkCallback (JNIEnv * env,
    jobject thiz, jlong handle, jstring name)
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
Java_org_nnsuite_nnstreamer_Pipeline_nativeRemoveSinkCallback (JNIEnv * env,
    jobject thiz, jlong handle, jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_sink_h sink;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s *);

  /* get handle from table */
  sink = (ml_pipeline_sink_h) nns_get_element_handle (pipe_info, element_name);
  if (sink) {
    nns_remove_element_handle (pipe_info, element_name);
    res = JNI_TRUE;
  }

  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return res;
}
