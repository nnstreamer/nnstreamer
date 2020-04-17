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
 * @file	nnstreamer-native-api.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nnstreamer-native.h"

/* nnstreamer plugins and sub-plugins declaration */
#if !defined (NNS_SINGLE_ONLY)
GST_PLUGIN_STATIC_DECLARE (nnstreamer);
GST_PLUGIN_STATIC_DECLARE (amcsrc);
extern void init_filter_cpp (void);
extern void init_filter_custom (void);
extern void init_filter_custom_easy (void);
extern void init_dv (void);
extern void init_bb (void);
extern void init_il (void);
extern void init_pose (void);
extern void init_is (void);
#endif

#if defined (ENABLE_TENSORFLOW_LITE)
extern void init_filter_tflite (void);
#endif
#if defined (ENABLE_SNAP)
extern void init_filter_snap (void);
#endif
#if defined (ENABLE_NNFW)
extern void init_filter_nnfw (void);
#endif

/**
 * @brief External function from GStreamer Android.
 */
extern void gst_android_init (JNIEnv * env, jobject context);

/**
 * @brief Global lock for native functions.
 */
G_LOCK_DEFINE_STATIC (nns_native_lock);

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
    switch (item->type) {
#if !defined (NNS_SINGLE_ONLY)
      case NNS_ELEMENT_TYPE_SRC:
        ml_pipeline_src_release_handle ((ml_pipeline_src_h) item->handle);
        break;
      case NNS_ELEMENT_TYPE_SINK:
        ml_pipeline_sink_unregister ((ml_pipeline_sink_h) item->handle);
        break;
      case NNS_ELEMENT_TYPE_VALVE:
        ml_pipeline_valve_release_handle ((ml_pipeline_valve_h) item->handle);
        break;
      case NNS_ELEMENT_TYPE_SWITCH_IN:
      case NNS_ELEMENT_TYPE_SWITCH_OUT:
        ml_pipeline_switch_release_handle ((ml_pipeline_switch_h) item->handle);
        break;
#endif
      default:
        nns_logw ("Given element type %d is unknown.", item->type);
        if (item->handle)
          g_free (item->handle);
        break;
    }

    g_free (item->name);
    g_free (item);
  }
}

/**
 * @brief Construct pipeline info.
 */
gpointer
nns_construct_pipe_info (JNIEnv * env, jobject thiz, gpointer handle,
    nns_pipe_type_e type)
{
  pipeline_info_s *pipe_info;
  jclass cls_data, cls_info;

  pipe_info = g_new0 (pipeline_info_s, 1);
  g_return_val_if_fail (pipe_info != NULL, NULL);

  pipe_info->pipeline_type = type;
  pipe_info->pipeline_handle = handle;
  pipe_info->element_handles =
      g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
      nns_free_element_data);
  g_mutex_init (&pipe_info->lock);

  (*env)->GetJavaVM (env, &pipe_info->jvm);
  g_assert (pipe_info->jvm);
  pthread_key_create (&pipe_info->jni_env, NULL);

  pipe_info->version = (*env)->GetVersion (env);
  pipe_info->instance = (*env)->NewGlobalRef (env, thiz);

  cls_data = (*env)->FindClass (env, "org/nnsuite/nnstreamer/TensorsData");
  pipe_info->cls_tensors_data = (*env)->NewGlobalRef (env, cls_data);
  (*env)->DeleteLocalRef (env, cls_data);

  cls_info = (*env)->FindClass (env, "org/nnsuite/nnstreamer/TensorsInfo");
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
  g_return_if_fail (pipe_info != NULL);

  g_mutex_lock (&pipe_info->lock);
  g_hash_table_destroy (pipe_info->element_handles);
  pipe_info->element_handles = NULL;
  g_mutex_unlock (&pipe_info->lock);

  switch (pipe_info->pipeline_type) {
#if !defined (NNS_SINGLE_ONLY)
    case NNS_PIPE_TYPE_PIPELINE:
      ml_pipeline_destroy (pipe_info->pipeline_handle);
      break;
    case NNS_PIPE_TYPE_CUSTOM:
      /**
       * Do nothing here (no handle to close).
       * The handle is filter-framework and it will be closed in customfilter-destroy function.
       */
      break;
#endif
    case NNS_PIPE_TYPE_SINGLE:
      ml_single_close (pipe_info->pipeline_handle);
      break;
    default:
      nns_logw ("Given pipe type %d is unknown.", pipe_info->pipeline_type);
      if (pipe_info->pipeline_handle)
        g_free (pipe_info->pipeline_handle);
      break;
  }

  g_mutex_clear (&pipe_info->lock);

  (*env)->DeleteGlobalRef (env, pipe_info->instance);
  (*env)->DeleteGlobalRef (env, pipe_info->cls_tensors_data);
  (*env)->DeleteGlobalRef (env, pipe_info->cls_tensors_info);

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
    ml_tensors_data_h data_h, ml_tensors_info_h info_h, jobject * result)
{
  guint i;
  jmethodID mid_init, mid_add_data, mid_set_info;
  jobject obj_data = NULL;
  ml_tensors_data_s *data;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (data_h, FALSE);
  g_return_val_if_fail (result, FALSE);

  data = (ml_tensors_data_s *) data_h;

  /* method to generate tensors data */
  mid_init =
      (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "<init>", "()V");
  mid_add_data =
      (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "addTensorData",
      "([B)V");
  mid_set_info =
      (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "setTensorsInfo",
      "(Lorg/nnsuite/nnstreamer/TensorsInfo;)V");

  obj_data = (*env)->NewObject (env, pipe_info->cls_tensors_data, mid_init);
  if (!obj_data) {
    nns_loge ("Failed to allocate object for tensors data.");
    goto done;
  }

  if (info_h) {
    jobject obj_info = NULL;

    if (!nns_convert_tensors_info (pipe_info, env, info_h, &obj_info)) {
      nns_loge ("Failed to convert tensors info.");
      (*env)->DeleteLocalRef (env, obj_data);
      obj_data = NULL;
      goto done;
    }

    (*env)->CallVoidMethod (env, obj_data, mid_set_info, obj_info);
    (*env)->DeleteLocalRef (env, obj_info);
  }

  for (i = 0; i < data->num_tensors; i++) {
    jsize buffer_size = (jsize) data->tensors[i].size;
    jbyteArray buffer = (*env)->NewByteArray (env, buffer_size);

    (*env)->SetByteArrayRegion (env, buffer, 0, buffer_size,
        (jbyte *) data->tensors[i].tensor);

    (*env)->CallVoidMethod (env, obj_data, mid_add_data, buffer);
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
    jobject obj_data, ml_tensors_data_h * data_h, ml_tensors_info_h * info_h)
{
  guint i;
  ml_tensors_data_s *data;
  ml_tensors_info_s *info;
  gboolean failed = FALSE;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (obj_data, FALSE);
  g_return_val_if_fail (data_h, FALSE);

  if (ml_tensors_data_create_no_alloc (NULL, data_h) != ML_ERROR_NONE) {
    nns_loge ("Failed to create handle for tensors data.");
    return FALSE;
  }

  data = (ml_tensors_data_s *) (*data_h);

  /* get field 'mDataList' */
  jfieldID fid_arraylist =
      (*env)->GetFieldID (env, pipe_info->cls_tensors_data, "mDataList",
      "Ljava/util/ArrayList;");
  jobject obj_arraylist = (*env)->GetObjectField (env, obj_data, fid_arraylist);

  /* method to get tensors data */
  jclass cls_arraylist = (*env)->GetObjectClass (env, obj_arraylist);
  jmethodID mid_size = (*env)->GetMethodID (env, cls_arraylist, "size", "()I");
  jmethodID mid_get =
      (*env)->GetMethodID (env, cls_arraylist, "get", "(I)Ljava/lang/Object;");

  /* number of tensors data */
  data->num_tensors =
      (unsigned int) (*env)->CallIntMethod (env, obj_arraylist, mid_size);

  /* set tensor data */
  for (i = 0; i < data->num_tensors; i++) {
    jobject tensor_data =
        (*env)->CallObjectMethod (env, obj_arraylist, mid_get, i);

    if (tensor_data) {
      size_t data_size =
          (size_t) (*env)->GetDirectBufferCapacity (env, tensor_data);
      gpointer data_ptr = (*env)->GetDirectBufferAddress (env, tensor_data);

      data->tensors[i].tensor = g_malloc (data_size);
      if (data->tensors[i].tensor == NULL) {
        nns_loge ("Failed to allocate memory %zd, data index %d.", data_size,
            i);
        (*env)->DeleteLocalRef (env, tensor_data);
        failed = TRUE;
        goto done;
      }

      memcpy (data->tensors[i].tensor, data_ptr, data_size);
      data->tensors[i].size = data_size;

      (*env)->DeleteLocalRef (env, tensor_data);
    }
  }

  /* parse tensors info in data class */
  if (info_h) {
    jmethodID mid_get_info =
        (*env)->GetMethodID (env, pipe_info->cls_tensors_data,
        "getTensorsInfo", "()Lorg/nnsuite/nnstreamer/TensorsInfo;");
    jobject obj_info = (*env)->CallObjectMethod (env, obj_data, mid_get_info);

    if (obj_info) {
      nns_parse_tensors_info (pipe_info, env, obj_info, info_h);
      (*env)->DeleteLocalRef (env, obj_info);
    }
  }

done:
  (*env)->DeleteLocalRef (env, cls_arraylist);
  (*env)->DeleteLocalRef (env, obj_arraylist);

  if (failed) {
    ml_tensors_data_destroy (*data_h);
    *data_h = NULL;
  }

  return !failed;
}

/**
 * @brief Convert tensors info to TensorsInfo object.
 */
gboolean
nns_convert_tensors_info (pipeline_info_s * pipe_info, JNIEnv * env,
    ml_tensors_info_h info_h, jobject * result)
{
  guint i, j;
  ml_tensors_info_s *info;
  jmethodID mid_init, mid_add;
  jobject obj_info = NULL;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (info_h, FALSE);
  g_return_val_if_fail (result, FALSE);

  info = (ml_tensors_info_s *) info_h;

  /* method to generate tensors info */
  mid_init =
      (*env)->GetMethodID (env, pipe_info->cls_tensors_info, "<init>", "()V");
  mid_add =
      (*env)->GetMethodID (env, pipe_info->cls_tensors_info, "appendInfo",
      "(Ljava/lang/String;I[I)V");

  obj_info = (*env)->NewObject (env, pipe_info->cls_tensors_info, mid_init);
  if (!obj_info) {
    nns_loge ("Failed to allocate object for tensors info.");
    goto done;
  }

  for (i = 0; i < info->num_tensors; i++) {
    jstring name = NULL;
    jint type;
    jintArray dimension;
    jint *dim;

    if (info->info[i].name)
      name = (*env)->NewStringUTF (env, info->info[i].name);

    type = (jint) info->info[i].type;

    dimension = (*env)->NewIntArray (env, ML_TENSOR_RANK_LIMIT);

    dim = (*env)->GetIntArrayElements (env, dimension, NULL);
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
    jobject obj_info, ml_tensors_info_h * info_h)
{
  guint i, j;
  ml_tensors_info_s *info;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (obj_info, FALSE);
  g_return_val_if_fail (info_h, FALSE);

  if (ml_tensors_info_create (info_h) != ML_ERROR_NONE) {
    nns_loge ("Failed to create handle for tensors info.");
    return FALSE;
  }

  info = (ml_tensors_info_s *) (*info_h);

  /* get field 'mInfoList' */
  jfieldID fid_arraylist =
      (*env)->GetFieldID (env, pipe_info->cls_tensors_info, "mInfoList",
      "Ljava/util/ArrayList;");
  jobject obj_arraylist = (*env)->GetObjectField (env, obj_info, fid_arraylist);

  /* method to get tensors info */
  jclass cls_arraylist = (*env)->GetObjectClass (env, obj_arraylist);
  jmethodID mid_size = (*env)->GetMethodID (env, cls_arraylist, "size", "()I");
  jmethodID mid_get =
      (*env)->GetMethodID (env, cls_arraylist, "get", "(I)Ljava/lang/Object;");

  /* number of tensors info */
  info->num_tensors =
      (unsigned int) (*env)->CallIntMethod (env, obj_arraylist, mid_size);

  /* read tensor info */
  for (i = 0; i < info->num_tensors; i++) {
    jobject item = (*env)->CallObjectMethod (env, obj_arraylist, mid_get, i);
    jclass cls_info = (*env)->GetObjectClass (env, item);

    jmethodID mid_name =
        (*env)->GetMethodID (env, cls_info, "getName", "()Ljava/lang/String;");
    jmethodID mid_type =
        (*env)->GetMethodID (env, cls_info, "getTypeValue", "()I");
    jmethodID mid_dimension =
        (*env)->GetMethodID (env, cls_info, "getDimension", "()[I");

    /* tensor name */
    jstring name_str = (jstring) (*env)->CallObjectMethod (env, item, mid_name);
    if (name_str) {
      const gchar *name = (*env)->GetStringUTFChars (env, name_str, NULL);

      info->info[i].name = g_strdup (name);
      (*env)->ReleaseStringUTFChars (env, name_str, name);
      (*env)->DeleteLocalRef (env, name_str);
    }

    /* tensor type */
    info->info[i].type =
        (ml_tensor_type_e) (*env)->CallIntMethod (env, item, mid_type);

    /* tensor dimension */
    jintArray dim =
        (jintArray) (*env)->CallObjectMethod (env, item, mid_dimension);
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
  }

  (*env)->DeleteLocalRef (env, cls_arraylist);
  (*env)->DeleteLocalRef (env, obj_arraylist);
  return TRUE;
}

/**
 * @brief Get NNFW from integer value.
 */
gboolean
nns_get_nnfw_type (jint fw_type, ml_nnfw_type_e * nnfw)
{
  gboolean is_supported = TRUE;

  if (!nnfw)
    return FALSE;

  *nnfw = ML_NNFW_TYPE_ANY;

  /* enumeration defined in NNStreamer.java */
  switch (fw_type) {
    case 0: /* NNFWType.TENSORFLOW_LITE */
      *nnfw = ML_NNFW_TYPE_TENSORFLOW_LITE;
#if !defined (ENABLE_TENSORFLOW_LITE)
      nns_logw ("tensorflow-lite is not supported.");
      is_supported = FALSE;
#endif
      break;
    case 1: /* NNFWType.SNAP */
      *nnfw = ML_NNFW_TYPE_SNAP;
#if !defined (ENABLE_SNAP)
      nns_logw ("SNAP is not supported.");
      is_supported = FALSE;
#endif
      break;
    case 2: /* NNFWType.NNFW */
      *nnfw = ML_NNFW_TYPE_NNFW;
#if !defined (ENABLE_NNFW)
      nns_logw ("NNFW is not supported.");
      is_supported = FALSE;
#endif
      break;
    default: /* Unknown */
      nns_logw ("Unknown NNFW type (%d).", fw_type);
      is_supported = FALSE;
      break;
  }

  return is_supported && ml_nnfw_is_available (*nnfw, ML_NNFW_HW_ANY);
}

/**
 * @brief Initialize NNStreamer, register required plugins.
 */
jboolean
nnstreamer_native_initialize (JNIEnv * env, jobject context)
{
  jboolean result = JNI_FALSE;
  static gboolean nns_is_initilaized = FALSE;

  nns_logi ("Called native initialize.");

  G_LOCK (nns_native_lock);

  if (!gst_is_initialized ())
    gst_android_init (env, context);

  if (!gst_is_initialized ()) {
    nns_loge ("GStreamer is not initialized.");
    goto done;
  }

  if (nns_is_initilaized == FALSE) {
    /* register nnstreamer plugins */
#if !defined (NNS_SINGLE_ONLY)
    GST_PLUGIN_STATIC_REGISTER (nnstreamer);

    /* Android MediaCodec */
    GST_PLUGIN_STATIC_REGISTER (amcsrc);

    /* tensor-filter sub-plugins */
    init_filter_cpp ();
    init_filter_custom ();
    init_filter_custom_easy ();

    /* tensor-decoder sub-plugins */
    init_dv ();
    init_bb ();
    init_il ();
    init_pose ();
    init_is ();
#endif

#if defined (ENABLE_TENSORFLOW_LITE)
    init_filter_tflite ();
#endif
#if defined (ENABLE_SNAP)
    init_filter_snap ();
#endif
#if defined (ENABLE_NNFW)
    init_filter_nnfw ();
#endif

    nns_is_initilaized = TRUE;
  }

  result = JNI_TRUE;

  /* print version info */
  gchar *gst_ver = gst_version_string ();
  gchar *nns_ver = nnstreamer_version_string ();

  nns_logi ("%s %s GLib %d.%d.%d", nns_ver, gst_ver, GLIB_MAJOR_VERSION,
      GLIB_MINOR_VERSION, GLIB_MICRO_VERSION);

  g_free (gst_ver);
  g_free (nns_ver);

done:
  G_UNLOCK (nns_native_lock);
  return result;
}

/**
 * @brief Native method to initialize NNStreamer.
 */
jboolean
Java_org_nnsuite_nnstreamer_NNStreamer_nativeInitialize (JNIEnv * env,
    jclass clazz, jobject context)
{
  return nnstreamer_native_initialize (env, context);
}

/**
 * @brief Native method to check the availability of NNFW.
 */
jboolean
Java_org_nnsuite_nnstreamer_NNStreamer_nativeCheckAvailability (JNIEnv * env,
    jclass clazz, jint fw_type)
{
  ml_nnfw_type_e nnfw;

  if (!nns_get_nnfw_type (fw_type, &nnfw)) {
    return JNI_FALSE;
  }

  return JNI_TRUE;
}

/**
 * @brief Native method to get the version string of NNStreamer.
 */
jstring
Java_org_nnsuite_nnstreamer_NNStreamer_nativeGetVersion (JNIEnv * env,
    jclass clazz)
{
  gchar *nns_ver = nnstreamer_version_string ();
  jstring version = (*env)->NewStringUTF (env, nns_ver);

  g_free (nns_ver);
  return version;
}
