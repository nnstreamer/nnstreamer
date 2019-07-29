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
 * @file	nnstreamer-native-api.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/**
 * @brief Attach thread with Java VM.
 */
static JNIEnv *
nns_attach_current_thread (pipeline_info_s * pipe_info)
{
  JNIEnv *env;
  JavaVM *jvm;
  JavaVMAttachArgs args;

  g_assert (pipe_info);
  jvm = pipe_info->jvm;

  args.version = pipe_info->version;
  args.name = NULL;
  args.group = NULL;

  if ((*jvm)->AttachCurrentThread (jvm, &env, &args) < 0) {
    nns_loge ("Failed to attach current thread.");
    return NULL;
  }

  return env;
}

/**
 * @brief Get JNI environment.
 */
JNIEnv *
nns_get_jni_env (pipeline_info_s * pipe_info)
{
  JNIEnv *env;

  g_assert (pipe_info);

  if ((env = pthread_getspecific (pipe_info->jni_env)) == NULL) {
    env = nns_attach_current_thread (pipe_info);
    pthread_setspecific (pipe_info->jni_env, env);
  }

  return env;
}

/**
 * @brief Free element handle pointer.
 */
void
nns_free_element_data (gpointer data)
{
  element_data_s *item = (element_data_s *) data;

  if (item) {
    if (g_str_equal (item->type, NNS_ELEMENT_TYPE_SRC)) {
      ml_pipeline_src_release_handle ((ml_pipeline_src_h) item->handle);
    } else if (g_str_equal (item->type, NNS_ELEMENT_TYPE_SINK)) {
      ml_pipeline_sink_unregister ((ml_pipeline_sink_h) item->handle);
    } else if (g_str_equal (item->type, NNS_ELEMENT_TYPE_SWITCH_IN) ||
        g_str_equal (item->type, NNS_ELEMENT_TYPE_SWITCH_OUT)) {
      ml_pipeline_switch_release_handle ((ml_pipeline_switch_h) item->handle);
    } else if (g_str_equal (item->type, NNS_ELEMENT_TYPE_VALVE)) {
      ml_pipeline_valve_release_handle ((ml_pipeline_valve_h) item->handle);
    } else {
      nns_logw ("Given element type %s is unknown.", item->type);
      if (item->handle)
        g_free (item->handle);
    }

    g_free (item->name);
    g_free (item->type);
    g_free (item);
  }
}

/**
 * @brief Construct pipeline info.
 */
gpointer
nns_construct_pipe_info (JNIEnv * env, jobject thiz, gpointer handle, const gchar * type)
{
  pipeline_info_s *pipe_info;

  pipe_info = g_new0 (pipeline_info_s, 1);
  g_assert (pipe_info);

  pipe_info->pipeline_type = g_strdup (type);
  pipe_info->pipeline_handle = handle;
  pipe_info->element_handles = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, nns_free_element_data);
  g_mutex_init (&pipe_info->lock);

  (*env)->GetJavaVM (env, &pipe_info->jvm);
  g_assert (pipe_info->jvm);
  pthread_key_create (&pipe_info->jni_env, NULL);

  pipe_info->version = (*env)->GetVersion (env);
  pipe_info->instance = (*env)->NewGlobalRef (env, thiz);

  jclass cls_data = (*env)->FindClass (env, "com/samsung/android/nnstreamer/TensorsData");
  pipe_info->cls_tensors_data = (*env)->NewGlobalRef (env, cls_data);
  (*env)->DeleteLocalRef (env, cls_data);

  jclass cls_info = (*env)->FindClass (env, "com/samsung/android/nnstreamer/TensorsInfo");
  pipe_info->cls_tensors_info = (*env)->NewGlobalRef (env, cls_info);
  (*env)->DeleteLocalRef (env, cls_info);

  return pipe_info;
}

/**
 * @brief Destroy pipeline info.
 */
void
nns_destroy_pipe_info (pipeline_info_s * pipe_info, JNIEnv * env)
{
  g_assert (pipe_info);

  g_mutex_lock (&pipe_info->lock);
  g_hash_table_destroy (pipe_info->element_handles);
  pipe_info->element_handles = NULL;
  g_mutex_unlock (&pipe_info->lock);

  if (g_str_equal (pipe_info->pipeline_type, NNS_PIPE_TYPE_PIPELINE)) {
    ml_pipeline_destroy (pipe_info->pipeline_handle);
  } else if (g_str_equal (pipe_info->pipeline_type, NNS_PIPE_TYPE_SINGLE)) {
    ml_single_close (pipe_info->pipeline_handle);
  } else if (g_str_equal (pipe_info->pipeline_type, NNS_PIPE_TYPE_CUSTOM)) {
    /**
     * Do nothing here (no handle to close).
     * The handle is filter-framework and it will be closed in customfilter-destroy function.
     */
  } else {
    nns_logw ("Given pipe type %s is unknown.", pipe_info->pipeline_type);
    if (pipe_info->pipeline_handle)
      g_free (pipe_info->pipeline_handle);
  }

  g_mutex_clear (&pipe_info->lock);

  (*env)->DeleteGlobalRef (env, pipe_info->instance);
  (*env)->DeleteGlobalRef (env, pipe_info->cls_tensors_data);
  (*env)->DeleteGlobalRef (env, pipe_info->cls_tensors_info);

  g_free (pipe_info->pipeline_type);
  g_free (pipe_info);
}

/**
 * @brief Get element handle of given name.
 */
gpointer
nns_get_element_handle (pipeline_info_s * pipe_info, const gchar * name)
{
  element_data_s *item;

  g_return_val_if_fail (pipe_info, NULL);
  g_return_val_if_fail (name, NULL);

  g_mutex_lock (&pipe_info->lock);
  item = g_hash_table_lookup (pipe_info->element_handles, name);
  g_mutex_unlock (&pipe_info->lock);

  return (item) ? item->handle : NULL;
}

/**
 * @brief Remove element handle of given name.
 */
gboolean
nns_remove_element_handle (pipeline_info_s * pipe_info, const gchar * name)
{
  gboolean ret;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (name, FALSE);

  g_mutex_lock (&pipe_info->lock);
  ret = g_hash_table_remove (pipe_info->element_handles, name);
  g_mutex_unlock (&pipe_info->lock);

  return ret;
}

/**
 * @brief Add new element handle of given name and type.
 */
gboolean
nns_add_element_handle (pipeline_info_s * pipe_info, const gchar * name,
    element_data_s * item)
{
  gboolean ret;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (name && item, FALSE);

  g_mutex_lock (&pipe_info->lock);
  ret = g_hash_table_insert (pipe_info->element_handles, g_strdup (name), item);
  g_mutex_unlock (&pipe_info->lock);

  return ret;
}

/**
 * @brief Convert tensors data to TensorsData object.
 */
gboolean
nns_convert_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env,
    ml_tensors_data_s * data, jobject * result)
{
  guint i;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (data && result, FALSE);

  /* method to generate tensors data */
  jmethodID mid_init = (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "<init>", "()V");
  jmethodID mid_add = (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "addTensorData", "([B)V");

  jobject obj_data = (*env)->NewObject (env, pipe_info->cls_tensors_data, mid_init);
  if (!obj_data) {
    nns_loge ("Failed to allocate object for tensors data.");
    goto done;
  }

  for (i = 0; i < data->num_tensors; i++) {
    jsize buffer_size = (jsize) data->tensors[i].size;
    jbyteArray buffer = (*env)->NewByteArray (env, buffer_size);

    (*env)->SetByteArrayRegion (env, buffer, 0, buffer_size, (jbyte *) data->tensors[i].tensor);

    (*env)->CallVoidMethod (env, obj_data, mid_add, buffer);
    (*env)->DeleteLocalRef (env, buffer);
  }

done:
  *result = obj_data;
  return (obj_data != NULL);
}

/**
 * @brief Parse tensors data from TensorsData object.
 */
gboolean
nns_parse_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env,
    jobject obj_data, ml_tensors_data_s * data)
{
  guint i;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (obj_data && data, FALSE);

  /* get field 'mDataList' */
  jfieldID fid_arraylist = (*env)->GetFieldID (env, pipe_info->cls_tensors_data, "mDataList", "java/util/ArrayList");
  jobject obj_arraylist = (*env)->GetObjectField (env, obj_data, fid_arraylist);

  /* method to get tensors data */
  jclass cls_arraylist = (*env)->GetObjectClass (env, obj_arraylist);
  jmethodID mid_size = (*env)->GetMethodID (env, cls_arraylist, "size", "()I");
  jmethodID mid_get = (*env)->GetMethodID (env, cls_arraylist, "get", "(I)Ljava/lang/Object;");

  /* number of tensors data */
  data->num_tensors = (unsigned int) (*env)->CallIntMethod (env, obj_arraylist, mid_size);

  /* set tensor data */
  for (i = 0; i < data->num_tensors; i++) {
    jobject tensor_data = (*env)->CallObjectMethod (env, obj_arraylist, mid_get, i);

    if (tensor_data) {
      size_t data_size = (size_t) (*env)->GetDirectBufferCapacity (env, tensor_data);
      gpointer data_ptr = (*env)->GetDirectBufferAddress (env, tensor_data);

      data->tensors[i].tensor = g_malloc (data_size);
      if (!data->tensors[i].tensor) {
        nns_loge ("Failed to allocate memory %zd, data index %d.", data_size, i);
        (*env)->DeleteLocalRef (env, tensor_data);
        goto failed;
      }

      memcpy (data->tensors[i].tensor, data_ptr, data_size);
      data->tensors[i].size = data_size;

      (*env)->DeleteLocalRef (env, tensor_data);
    }

    print_log ("Parse tensors data [%d] data at %p size %zd",
        i, data->tensors[i].tensor, data->tensors[i].size);
  }

  (*env)->DeleteLocalRef (env, cls_arraylist);
  (*env)->DeleteLocalRef (env, obj_arraylist);
  return TRUE;

failed:
  for (i = 0; i < data->num_tensors; i++) {
    if (data->tensors[i].tensor) {
      g_free (data->tensors[i].tensor);
      data->tensors[i].tensor = NULL;
    }

    data->tensors[i].size = 0;
  }

  data->num_tensors = 0;
  return FALSE;
}

/**
 * @brief Convert tensors info to TensorsInfo object.
 */
gboolean
nns_convert_tensors_info (pipeline_info_s * pipe_info, JNIEnv * env,
    ml_tensors_info_s * info, jobject * result)
{
  guint i, j;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (info && result, FALSE);

  /* method to generate tensors info */
  jmethodID mid_init = (*env)->GetMethodID (env, pipe_info->cls_tensors_info, "<init>", "()V");
  jmethodID mid_add = (*env)->GetMethodID (env, pipe_info->cls_tensors_info, "addTensorInfo", "(Ljava/lang/String;I[I)V");

  jobject obj_info = (*env)->NewObject (env, pipe_info->cls_tensors_info, mid_init);
  if (!obj_info) {
    nns_loge ("Failed to allocate object for tensors info.");
    goto done;
  }

  for (i = 0; i < info->num_tensors; i++) {
    jstring name = NULL;
    jint type;
    jintArray dimension;

    if (info->info[i].name)
      name = (*env)->NewStringUTF (env, info->info[i].name);

    type = (jint) info->info[i].type;

    dimension = (*env)->NewIntArray (env, ML_TENSOR_RANK_LIMIT);

    jint *dim = (*env)->GetIntArrayElements (env, dimension, NULL);
    for (j = 0; j < ML_TENSOR_RANK_LIMIT; j++) {
      dim[j] = (jint) info->info[i].dimension[j];
    }
    (*env)->ReleaseIntArrayElements (env, dimension, dim, 0);

    (*env)->CallVoidMethod (env, obj_info, mid_add, name, type, dimension);

    if (name)
      (*env)->DeleteLocalRef (env, name);
    (*env)->DeleteLocalRef (env, dimension);
  }

done:
  *result = obj_info;
  return (obj_info != NULL);
}

/**
 * @brief Parse tensors info from TensorsInfo object.
 */
gboolean
nns_parse_tensors_info (pipeline_info_s * pipe_info, JNIEnv * env,
    jobject obj_info, ml_tensors_info_s * info)
{
  guint i, j;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (obj_info && info, FALSE);

  ml_tensors_info_initialize (info);

  /* get field 'mInfoList' */
  jfieldID fid_arraylist = (*env)->GetFieldID (env, pipe_info->cls_tensors_info, "mInfoList", "java/util/ArrayList");
  jobject obj_arraylist = (*env)->GetObjectField (env, obj_info, fid_arraylist);

  /* method to get tensors info */
  jclass cls_arraylist = (*env)->GetObjectClass (env, obj_arraylist);
  jmethodID mid_size = (*env)->GetMethodID (env, cls_arraylist, "size", "()I");
  jmethodID mid_get = (*env)->GetMethodID (env, cls_arraylist, "get", "(I)Ljava/lang/Object;");

  /* number of tensors info */
  info->num_tensors = (unsigned int) (*env)->CallIntMethod (env, obj_arraylist, mid_size);

  /* read tensor info */
  for (i = 0; i < info->num_tensors; i++) {
    jobject item = (*env)->CallObjectMethod (env, obj_arraylist, mid_get, i);
    jclass cls_info = (*env)->GetObjectClass (env, item);

    jmethodID mid_name = (*env)->GetMethodID (env, cls_info, "getName", "()Ljava/lang/String;");
    jmethodID mid_type = (*env)->GetMethodID (env, cls_info, "getType", "()I");
    jmethodID mid_dimension = (*env)->GetMethodID (env, cls_info, "getDimension", "()[I");

    /* tensor name */
    jstring name_str = (jstring) (*env)->CallObjectMethod (env, item, mid_name);
    if (name_str) {
      const gchar *name = (*env)->GetStringUTFChars (env, name_str, NULL);

      info->info[i].name = g_strdup (name);
      (*env)->ReleaseStringUTFChars (env, name_str, name);
      (*env)->DeleteLocalRef (env, name_str);
    }

    /* tensor type */
    info->info[i].type = (ml_tensor_type_e) (*env)->CallIntMethod (env, item, mid_type);

    /* tensor dimension */
    jintArray dim = (jintArray) (*env)->CallObjectMethod (env, item, mid_dimension);
    jsize length = (*env)->GetArrayLength (env, dim);
    jint *dimension = (*env)->GetIntArrayElements (env, dim, NULL);

    g_assert (length == ML_TENSOR_RANK_LIMIT);
    for (j = 0; j < length; j++) {
      info->info[i].dimension[j] = (unsigned int) dimension[j];
    }

    (*env)->ReleaseIntArrayElements (env, dim, dimension, 0);
    (*env)->DeleteLocalRef (env, dim);

    (*env)->DeleteLocalRef (env, cls_info);
    (*env)->DeleteLocalRef (env, item);

    print_log ("Parse tensors info [%d] name %s type %d dim %d:%d:%d:%d",
        i, GST_STR_NULL (info->info[i].name), info->info[i].type,
        info->info[i].dimension[0], info->info[i].dimension[1],
        info->info[i].dimension[2], info->info[i].dimension[3]);
  }

  (*env)->DeleteLocalRef (env, cls_arraylist);
  (*env)->DeleteLocalRef (env, obj_arraylist);
  return TRUE;
}

/**
 * @brief Native method to initialize NNStreamer.
 */
jboolean
Java_com_samsung_android_nnstreamer_NNStreamer_nativeInitialize (JNIEnv * env, jclass clazz,
    jobject context)
{
  nns_logi ("Called native initialize.");

  if (!gst_is_initialized ()) {
    nns_loge ("GStreamer is not initialized.");
    return JNI_FALSE;
  }

  /* register nnstreamer plugins */
  GST_PLUGIN_STATIC_REGISTER (nnstreamer);

  /* filter tensorflow-lite sub-plugin */
  init_filter_tflite ();

  /* decoder sub-plugins */
  init_dv ();
  init_bb ();
  init_il ();
  init_pose ();

  return JNI_TRUE;
}

/**
 * @brief Native method to get the version string of NNStreamer and GStreamer.
 */
jstring
Java_com_samsung_android_nnstreamer_NNStreamer_nativeGetVersion (JNIEnv * env, jclass clazz)
{
  gchar *gst_ver = gst_version_string ();
  gchar *version_str = g_strdup_printf ("NNStreamer %s, %s", VERSION, gst_ver);

  jstring version = (*env)->NewStringUTF (env, version_str);

  g_free (gst_ver);
  g_free (version_str);
  return version;
}
