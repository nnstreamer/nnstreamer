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
 * @file	nnstreamer-native.h
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_ANDROID_NATIVE_H__
#define __NNSTREAMER_ANDROID_NATIVE_H__

#include <jni.h>

#include <gst/gst.h>

#include "nnstreamer.h"
#include "nnstreamer-single.h"
#include "nnstreamer-capi-private.h"
#include "nnstreamer_log.h"
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_filter.h"

#if GLIB_SIZEOF_VOID_P == 8
#define CAST_TO_LONG(p) (jlong)(p)
#define CAST_TO_TYPE(l,type) (type)(l)
#else
#define CAST_TO_LONG(p) (jlong)(jint)(p)
#define CAST_TO_TYPE(l,type) (type)(jint)(l)
#endif

/**
 * @brief NNStreamer package name.
 */
#define NNS_PKG "org/nnsuite/nnstreamer"
#define NNS_CLS_TDATA NNS_PKG "/TensorsData"
#define NNS_CLS_TINFO NNS_PKG "/TensorsInfo"
#define NNS_CLS_PIPELINE NNS_PKG "/Pipeline"
#define NNS_CLS_SINGLE NNS_PKG "/SingleShot"
#define NNS_CLS_CUSTOM_FILTER NNS_PKG "/CustomFilter"
#define NNS_CLS_NNSTREAMER NNS_PKG "/NNStreamer"

/**
 * @brief Callback to destroy private data in pipe info.
 */
typedef void (*nns_priv_destroy)(gpointer data, JNIEnv * env);

/**
 * @brief Pipeline type in native pipe info.
 */
typedef enum
{
  NNS_PIPE_TYPE_PIPELINE = 0,
  NNS_PIPE_TYPE_SINGLE,
  NNS_PIPE_TYPE_CUSTOM,

  NNS_PIPE_TYPE_UNKNOWN
} nns_pipe_type_e;

/**
 * @brief Element type in native pipe info.
 */
typedef enum
{
  NNS_ELEMENT_TYPE_SRC = 0,
  NNS_ELEMENT_TYPE_SINK,
  NNS_ELEMENT_TYPE_VALVE,
  NNS_ELEMENT_TYPE_SWITCH,

  NNS_ELEMENT_TYPE_UNKNOWN
} nns_element_type_e;

/**
 * @brief Struct for TensorsData class info.
 */
typedef struct
{
  jclass cls;
  jmethodID mid_init;
  jmethodID mid_alloc;
  jmethodID mid_get_array;
  jmethodID mid_get_info;
} data_class_info_s;

/**
 * @brief Struct for TensorsInfo class info.
 */
typedef struct
{
  jclass cls;
  jmethodID mid_init;
  jmethodID mid_add_info;
  jmethodID mid_get_array;

  jclass cls_info;
  jfieldID fid_info_name;
  jfieldID fid_info_type;
  jfieldID fid_info_dim;
} info_class_info_s;

/**
 * @brief Struct for constructed pipeline.
 */
typedef struct
{
  nns_pipe_type_e pipeline_type;
  gpointer pipeline_handle;
  GHashTable *element_handles;
  GMutex lock;

  JavaVM *jvm;
  jint version;
  pthread_key_t jni_env;

  jobject instance;
  jclass cls;
  data_class_info_s data_cls_info;
  info_class_info_s info_cls_info;

  gpointer priv_data;
  nns_priv_destroy priv_destroy_func;
} pipeline_info_s;

/**
 * @brief Struct for element data in pipeline.
 */
typedef struct
{
  gchar *name;
  nns_element_type_e type;
  gpointer handle;
  pipeline_info_s *pipe_info;

  gpointer priv_data;
  nns_priv_destroy priv_destroy_func;
} element_data_s;

/**
 * @brief Get JNI environment.
 */
extern JNIEnv *
nns_get_jni_env (pipeline_info_s * pipe_info);

/**
 * @brief Free element handle pointer.
 */
extern void
nns_free_element_data (gpointer data);

/**
 * @brief Construct pipeline info.
 */
extern gpointer
nns_construct_pipe_info (JNIEnv * env, jobject thiz, gpointer handle, nns_pipe_type_e type);

/**
 * @brief Destroy pipeline info.
 */
extern void
nns_destroy_pipe_info (pipeline_info_s * pipe_info, JNIEnv * env);

/**
 * @brief Set private data in pipeline info.
 */
extern void
nns_set_priv_data (pipeline_info_s * pipe_info, gpointer data, nns_priv_destroy destroy_func);

/**
 * @brief Get element handle of given name and type.
 */
extern gpointer
nns_get_element_handle (pipeline_info_s * pipe_info, const gchar * name, const nns_element_type_e type);

/**
 * @brief Remove element handle of given name.
 */
extern gboolean
nns_remove_element_handle (pipeline_info_s * pipe_info, const gchar * name);

/**
 * @brief Add new element handle of given name and type.
 */
extern gboolean
nns_add_element_handle (pipeline_info_s * pipe_info, const gchar * name, element_data_s * item);

/**
 * @brief Create new data object with given tensors info. Caller should unref the result object.
 */
extern gboolean
nns_create_tensors_data_object (pipeline_info_s * pipe_info, JNIEnv * env, jobject obj_info, jobject * result);

/**
 * @brief Convert tensors data to TensorsData object.
 */
extern gboolean
nns_convert_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env, ml_tensors_data_h data_h, jobject obj_info, jobject * result);

/**
 * @brief Parse tensors data from TensorsData object.
 */
extern gboolean
nns_parse_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env, jobject obj_data, gboolean clone, ml_tensors_data_h * data_h, ml_tensors_info_h * info_h);

/**
 * @brief Convert tensors info to TensorsInfo object.
 */
extern gboolean
nns_convert_tensors_info (pipeline_info_s * pipe_info, JNIEnv * env, ml_tensors_info_h info_h, jobject * result);

/**
 * @brief Parse tensors info from TensorsInfo object.
 */
extern gboolean
nns_parse_tensors_info (pipeline_info_s * pipe_info, JNIEnv * env, jobject obj_info, ml_tensors_info_h * info_h);

/**
 * @brief Get NNFW from integer value.
 */
extern gboolean
nns_get_nnfw_type (jint fw_type, ml_nnfw_type_e * nnfw);

/* Below defines native methods for each class */
extern jlong
nns_native_single_open (JNIEnv * env, jobject thiz, jobjectArray models, jobject in, jobject out, jint fw_type, jstring option);
extern void
nns_native_single_close (JNIEnv * env, jobject thiz, jlong handle);
extern jobject
nns_native_single_invoke (JNIEnv * env, jobject thiz, jlong handle, jobject in);
extern jobject
nns_native_single_get_input_info (JNIEnv * env, jobject thiz, jlong handle);
extern jobject
nns_native_single_get_output_info (JNIEnv * env, jobject thiz, jlong handle);
extern jboolean
nns_native_single_set_prop (JNIEnv * env, jobject thiz, jlong handle, jstring name, jstring value);
extern jstring
nns_native_single_get_prop (JNIEnv * env, jobject thiz, jlong handle, jstring name);
extern jboolean
nns_native_single_set_timeout (JNIEnv * env, jobject thiz, jlong handle, jint timeout);
extern jboolean
nns_native_single_set_input_info (JNIEnv * env, jobject thiz, jlong handle, jobject in);
#if !defined (NNS_SINGLE_ONLY)
extern jlong
nns_native_custom_initialize (JNIEnv * env, jobject thiz, jstring name, jobject in, jobject out);
extern void
nns_native_custom_destroy (JNIEnv * env, jobject thiz, jlong handle);
extern jlong
nns_native_pipe_construct (JNIEnv * env, jobject thiz, jstring description, jboolean add_state_cb);
extern void
nns_native_pipe_destroy (JNIEnv * env, jobject thiz, jlong handle);
extern jboolean
nns_native_pipe_start (JNIEnv * env, jobject thiz, jlong handle);
extern jboolean
nns_native_pipe_stop (JNIEnv * env, jobject thiz, jlong handle);
extern jint
nns_native_pipe_get_state (JNIEnv * env, jobject thiz, jlong handle);
extern jboolean
nns_native_pipe_input_data (JNIEnv * env, jobject thiz, jlong handle, jstring name, jobject in);
extern jobjectArray
nns_native_pipe_get_switch_pads (JNIEnv * env, jobject thiz, jlong handle, jstring name);
extern jboolean
nns_native_pipe_select_switch_pad (JNIEnv * env, jobject thiz, jlong handle, jstring name, jstring pad);
extern jboolean
nns_native_pipe_control_valve (JNIEnv * env, jobject thiz, jlong handle, jstring name, jboolean open);
extern jboolean
nns_native_pipe_add_sink_cb (JNIEnv * env, jobject thiz, jlong handle, jstring name);
extern jboolean
nns_native_pipe_remove_sink_cb (JNIEnv * env, jobject thiz, jlong handle, jstring name);
#endif

#endif /* __NNSTREAMER_ANDROID_NATIVE_H__ */
