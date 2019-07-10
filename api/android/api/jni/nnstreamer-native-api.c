/**
 * @file	nnstreamer-native-api.c
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <jni.h>
#include <android/log.h>

#include <gst/gst.h>

#include "nnstreamer.h"
#include "nnstreamer-single.h"
#include "nnstreamer-capi-private.h"

#ifndef DBG
#define DBG FALSE
#endif

#define TAG "NNStreamer-native"

#define nns_logi(...) \
    __android_log_print (ANDROID_LOG_INFO, TAG, __VA_ARGS__)

#define nns_logw(...) \
    __android_log_print (ANDROID_LOG_WARN, TAG, __VA_ARGS__)

#define nns_loge(...) \
    __android_log_print (ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define nns_logd(...) \
    __android_log_print (ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

#if (DBG)
#define print_log nns_logd
#else
#define print_log(...)
#endif

#if GLIB_SIZEOF_VOID_P == 8
#define CAST_TO_LONG(p) (jlong)(p)
#define CAST_TO_TYPE(l,type) (type)(l)
#else
#define CAST_TO_LONG(p) (jlong)(jint)(p)
#define CAST_TO_TYPE(l,type) (type)(jint)(l)
#endif

#define NNS_PIPE_TYPE_PIPELINE "pipeline"
#define NNS_PIPE_TYPE_SINGLE "single"

#define NNS_ELEMENT_TYPE_SRC "src"
#define NNS_ELEMENT_TYPE_SINK "sink"
#define NNS_ELEMENT_TYPE_VALVE "valve"
#define NNS_ELEMENT_TYPE_SWITCH_IN "switch_in"
#define NNS_ELEMENT_TYPE_SWITCH_OUT "switch_out"

/**
 * @brief Struct for constructed pipeline.
 */
typedef struct
{
  gchar *pipeline_type;
  gpointer pipeline_handle;
  GHashTable *element_handles;
  GMutex lock;

  JavaVM *jvm;
  jint version;
  pthread_key_t jni_env;

  jobject instance;
  jclass cls_tensors_data;
  jclass cls_tensors_info;
} pipeline_info_s;

/**
 * @brief Struct for element data in pipeline.
 */
typedef struct
{
  gchar *name;
  gchar *type;
  gpointer handle;
  pipeline_info_s *pipe_info;
} element_data_s;

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
static JNIEnv *
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
static void
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
static gpointer
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
static void
nns_destroy_pipe_info (pipeline_info_s * pipe_info, JNIEnv * env)
{
  g_assert (pipe_info);

  if (g_str_equal (pipe_info->pipeline_type, NNS_PIPE_TYPE_PIPELINE)) {
    ml_pipeline_destroy (pipe_info->pipeline_handle);
  } else if (g_str_equal (pipe_info->pipeline_type, NNS_PIPE_TYPE_SINGLE)) {
    ml_single_close (pipe_info->pipeline_handle);
  } else {
    nns_logw ("Given pipe type %s is unknown.", pipe_info->pipeline_type);
    g_free (pipe_info->pipeline_handle);
  }

  g_mutex_lock (&pipe_info->lock);
  g_hash_table_destroy (pipe_info->element_handles);
  pipe_info->element_handles = NULL;
  g_mutex_unlock (&pipe_info->lock);

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
static gpointer
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
static gboolean
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
static gboolean
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
static gboolean
nns_convert_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env,
    ml_tensors_data_s * data, jobject * result)
{
  guint i;

  g_return_val_if_fail (pipe_info, FALSE);
  g_return_val_if_fail (env, FALSE);
  g_return_val_if_fail (data && result, FALSE);

  /* method to generate tensors data */
  jmethodID mid_init = (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "<init>", "()V");
  jmethodID mid_add = (*env)->GetMethodID (env, pipe_info->cls_tensors_data, "addTensorData", "(Ljava/lang/Object;)V");

  jobject obj_data = (*env)->NewObject (env, pipe_info->cls_tensors_data, mid_init);
  if (!obj_data) {
    nns_loge ("Failed to allocate object for tensors data.");
    goto done;
  }

  for (i = 0; i < data->num_tensors; i++) {
    jobject item = (*env)->NewDirectByteBuffer (env, data->tensors[i].tensor,
        (jlong) data->tensors[i].size);

    (*env)->CallVoidMethod (env, obj_data, mid_add, item);
    (*env)->DeleteLocalRef (env, item);
  }

done:
  *result = obj_data;
  return (obj_data != NULL);
}

/**
 * @brief Parse tensors data from TensorsData object.
 */
static gboolean
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
}

/**
 * @brief Convert tensors info to TensorsInfo object.
 */
static gboolean
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
static gboolean
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
  jmethodID mid_callback = (*env)->GetMethodID (env, cls_pipeline, "stateChanged", "(I)V");
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
nns_sink_data_cb (const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data)
{
  element_data_s *cb_data;
  pipeline_info_s *pipe_info;
  ml_tensors_data_s *out_data;
  ml_tensors_info_s *out_info;

  cb_data = (element_data_s *) user_data;
  pipe_info = cb_data->pipe_info;
  out_data = (ml_tensors_data_s *) data;
  out_info = (ml_tensors_info_s *) info;

  print_log ("Received new data from %s (total %d tensors)",
      cb_data->name, out_data->num_tensors);

  JNIEnv *env = nns_get_jni_env (pipe_info);
  if (env == NULL) {
    nns_logw ("Cannot get jni env in the sink callback.");
    return;
  }

  jobject obj_data, obj_info;

  obj_data = obj_info = NULL;
  if (nns_convert_tensors_data (pipe_info, env, out_data, &obj_data) &&
      nns_convert_tensors_info (pipe_info, env, out_info, &obj_info)) {
    /* method for sink callback */
    jclass cls_pipeline = (*env)->GetObjectClass (env, pipe_info->instance);
    jmethodID mid_callback = (*env)->GetMethodID (env, cls_pipeline, "newDataReceived",
        "(Ljava/lang/String;Lcom/samsung/android/nnstreamer/TensorsData;Lcom/samsung/android/nnstreamer/TensorsInfo;)V");
    jstring sink_name = (*env)->NewStringUTF (env, cb_data->name);

    (*env)->CallVoidMethod (env, pipe_info->instance, mid_callback, sink_name, obj_data, obj_info);

    if ((*env)->ExceptionCheck (env)) {
      nns_loge ("Failed to call the callback method.");
      (*env)->ExceptionClear (env);
    }

    (*env)->DeleteLocalRef (env, sink_name);
    (*env)->DeleteLocalRef (env, cls_pipeline);
  } else {
    nns_loge ("Failed to convert the result to data object.");
  }

  if (obj_data)
    (*env)->DeleteLocalRef (env, obj_data);
  if (obj_info)
    (*env)->DeleteLocalRef (env, obj_info);
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

  handle = (ml_pipeline_sink_h) nns_get_element_handle (pipe_info, element_name);
  if (handle == NULL) {
    /* get sink handle and register to table */
    element_data_s *item = g_new0 (element_data_s, 1);
    g_assert (item);

    status = ml_pipeline_sink_register (pipe, element_name, nns_sink_data_cb, item, &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get sink node %s.", element_name);
      g_free (item);
      return NULL;
    }

    item->name = g_strdup (element_name);
    item->type = g_strdup (NNS_ELEMENT_TYPE_SINK);
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
    g_assert (item);

    item->name = g_strdup (element_name);
    item->type = g_strdup (NNS_ELEMENT_TYPE_SRC);
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

  handle = (ml_pipeline_switch_h) nns_get_element_handle (pipe_info, element_name);
  if (handle == NULL) {
    /* get switch handle and register to table */
    status = ml_pipeline_switch_get_handle (pipe, element_name, &switch_type, &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get switch %s.", element_name);
      return NULL;
    }

    element_data_s *item = g_new0 (element_data_s, 1);
    g_assert (item);

    item->name = g_strdup (element_name);
    if (switch_type == ML_PIPELINE_SWITCH_INPUT_SELECTOR)
      item->type = g_strdup (NNS_ELEMENT_TYPE_SWITCH_IN);
    else
      item->type = g_strdup (NNS_ELEMENT_TYPE_SWITCH_OUT);
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

  handle = (ml_pipeline_valve_h) nns_get_element_handle (pipe_info, element_name);
  if (handle == NULL) {
    /* get valve handle and register to table */
    status = ml_pipeline_valve_get_handle (pipe, element_name, &handle);
    if (status != ML_ERROR_NONE) {
      nns_loge ("Failed to get valve %s.", element_name);
      return NULL;
    }

    element_data_s *item = g_new0 (element_data_s, 1);
    g_assert (item);

    item->name = g_strdup (element_name);
    item->type = g_strdup (NNS_ELEMENT_TYPE_VALVE);
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
 * @brief Native method for single-shot API.
 */
jlong
Java_com_samsung_android_nnstreamer_SingleShot_nativeOpen (JNIEnv * env, jobject thiz,
    jstring model, jobject in, jobject out)
{
  pipeline_info_s *pipe_info = NULL;
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  int status;
  const char *model_info = (*env)->GetStringUTFChars (env, model, NULL);

  single = in_info = out_info = NULL;

  if (in) {
    ml_tensors_info_create (&in_info);
    nns_parse_tensors_info (pipe_info, env, in, (ml_tensors_info_s *) in_info);
  }

  if (out) {
    ml_tensors_info_create (&out_info);
    nns_parse_tensors_info (pipe_info, env, out, (ml_tensors_info_s *) out_info);
  }

  /* supposed tensorflow-lite only for android */
  status = ml_single_open (&single, model_info, in_info, out_info,
      ML_NNFW_TYPE_ANY, ML_NNFW_HW_AUTO);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to create the pipeline.");
    goto done;
  }

  pipe_info = nns_construct_pipe_info (env, thiz, single, NNS_PIPE_TYPE_SINGLE);

done:
  ml_tensors_info_destroy (in_info);
  ml_tensors_info_destroy (out_info);

  (*env)->ReleaseStringUTFChars (env, model, model_info);
  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for single-shot API.
 */
void
Java_com_samsung_android_nnstreamer_SingleShot_nativeClose (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_com_samsung_android_nnstreamer_SingleShot_nativeInvoke (JNIEnv * env, jobject thiz,
    jlong handle, jobject in)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_data_s *input, *output;
  int status;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;
  output = NULL;

  input = g_new0 (ml_tensors_data_s, 1);
  if (!input) {
    nns_loge ("Failed to allocate memory for input data.");
    goto done;
  }

  if (!nns_parse_tensors_data (pipe_info, env, in, input)) {
    nns_loge ("Failed to parse input data.");
    goto done;
  }

  status = ml_single_invoke (single, input, (ml_tensors_data_h *) &output);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to get the result from pipeline.");
    goto done;
  }

  if (!nns_convert_tensors_data (pipe_info, env, output, &result)) {
    nns_loge ("Failed to convert the result to data.");
    result = NULL;
  }

done:
  ml_tensors_data_destroy ((ml_tensors_data_h) input);
  ml_tensors_data_destroy ((ml_tensors_data_h) output);
  return result;
}

/**
 * @brief Native method for single-shot API.
 */
jobject
Java_com_samsung_android_nnstreamer_SingleShot_nativeGetInputInfo (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h info;
  int status;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;

  status = ml_single_get_input_info (single, &info);
  if (status != ML_ERROR_NONE) {
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
Java_com_samsung_android_nnstreamer_SingleShot_nativeGetOutputInfo (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info;
  ml_single_h single;
  ml_tensors_info_h info;
  int status;
  jobject result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
  single = pipe_info->pipeline_handle;

  status = ml_single_get_output_info (single, &info);
  if (status != ML_ERROR_NONE) {
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
 * @brief Native method for pipeline API.
 */
jlong
Java_com_samsung_android_nnstreamer_Pipeline_nativeConstruct (JNIEnv * env, jobject thiz,
    jstring description, jboolean add_state_cb)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  int status;
  const char *pipeline = (*env)->GetStringUTFChars (env, description, NULL);

  print_log ("Pipeline: %s", pipeline);
  pipe_info = nns_construct_pipe_info (env, thiz, NULL, NNS_PIPE_TYPE_PIPELINE);

  if (add_state_cb)
    status = ml_pipeline_construct (pipeline, nns_pipeline_state_cb, pipe_info, &pipe);
  else
    status = ml_pipeline_construct (pipeline, NULL, NULL, &pipe);

  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to create the pipeline.");
    nns_destroy_pipe_info (pipe_info, env);
    pipe_info = NULL;
  } else {
    pipe_info->pipeline_handle = pipe;
  }

  (*env)->ReleaseStringUTFChars (env, description, pipeline);
  return CAST_TO_LONG (pipe_info);
}

/**
 * @brief Native method for pipeline API.
 */
void
Java_com_samsung_android_nnstreamer_Pipeline_nativeDestroy (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

  nns_destroy_pipe_info (pipe_info, env);
}

/**
 * @brief Native method for pipeline API.
 */
jboolean
Java_com_samsung_android_nnstreamer_Pipeline_nativeStart (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  int status;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
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
Java_com_samsung_android_nnstreamer_Pipeline_nativeStop (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  int status;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
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
Java_com_samsung_android_nnstreamer_Pipeline_nativeGetState (JNIEnv * env, jobject thiz,
    jlong handle)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_h pipe;
  ml_pipeline_state_e state;
  int status;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);
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
Java_com_samsung_android_nnstreamer_Pipeline_nativeInputData (JNIEnv * env, jobject thiz,
    jlong handle, jstring name, jobject in)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_src_h src;
  ml_tensors_data_s *input = NULL;
  int status;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

  src = (ml_pipeline_src_h) nns_get_src_handle (pipe_info, element_name);
  if (src == NULL) {
    goto done;
  }

  input = g_new0 (ml_tensors_data_s, 1);
  if (!input) {
    nns_loge ("Failed to allocate memory for input data.");
    goto done;
  }

  if (!nns_parse_tensors_data (pipe_info, env, in, input)) {
    nns_loge ("Failed to parse input data.");
    ml_tensors_data_destroy ((ml_tensors_data_h) input);
    goto done;
  }

  status = ml_pipeline_src_input_data (src, (ml_tensors_data_h) input,
      ML_PIPELINE_BUF_POLICY_AUTO_FREE);
  if (status != ML_ERROR_NONE) {
    nns_loge ("Failed to input tensors data.");
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
Java_com_samsung_android_nnstreamer_Pipeline_nativeGetSwitchPads (JNIEnv * env, jobject thiz,
    jlong handle, jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_switch_h node;
  int status;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);
  char **pad_list = NULL;
  guint i, total = 0;
  jobjectArray result = NULL;

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

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

    result = (*env)->NewObjectArray (env, total, cls_string, (*env)->NewStringUTF (env, ""));
    if (result == NULL) {
      nns_loge ("Failed to allocate string array.");
      goto done;
    }

    for (i = 0; i < total; i++) {
      (*env)->SetObjectArrayElement (env, result, i, (*env)->NewStringUTF (env, pad_list[i]));
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
Java_com_samsung_android_nnstreamer_Pipeline_nativeSelectSwitchPad (JNIEnv * env, jobject thiz,
    jlong handle, jstring name, jstring pad)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_switch_h node;
  int status;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);
  const char *pad_name = (*env)->GetStringUTFChars (env, pad, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

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
Java_com_samsung_android_nnstreamer_Pipeline_nativeControlValve (JNIEnv * env, jobject thiz,
    jlong handle, jstring name, jboolean open)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_valve_h node;
  int status;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

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
Java_com_samsung_android_nnstreamer_Pipeline_nativeAddSinkCallback (JNIEnv * env, jobject thiz,
    jlong handle, jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_sink_h sink;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

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
Java_com_samsung_android_nnstreamer_Pipeline_nativeRemoveSinkCallback (JNIEnv * env, jobject thiz,
    jlong handle, jstring name)
{
  pipeline_info_s *pipe_info = NULL;
  ml_pipeline_sink_h sink;
  jboolean res = JNI_FALSE;
  const char *element_name = (*env)->GetStringUTFChars (env, name, NULL);

  pipe_info = CAST_TO_TYPE (handle, pipeline_info_s*);

  /* get handle from table */
  sink = (ml_pipeline_sink_h) nns_get_element_handle (pipe_info, element_name);
  if (sink) {
    nns_remove_element_handle (pipe_info, element_name);
    res = JNI_TRUE;
  }

  (*env)->ReleaseStringUTFChars (env, name, element_name);
  return res;
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
