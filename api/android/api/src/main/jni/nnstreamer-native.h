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
 * @file	nnstreamer-native.h
 * @date	10 July 2019
 * @brief	Native code for NNStreamer API
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_ANDROID_NATIVE_H__
#define __NNSTREAMER_ANDROID_NATIVE_H__

#include <jni.h>
#include <android/log.h>

#include <gst/gst.h>

#include "nnstreamer.h"
#include "nnstreamer-single.h"
#include "nnstreamer-capi-private.h"
#include "nnstreamer_plugin_api.h"
#include "nnstreamer_plugin_api_filter.h"

#define TAG "NNStreamer-native"

#define nns_logi(...) \
    __android_log_print (ANDROID_LOG_INFO, TAG, __VA_ARGS__)

#define nns_logw(...) \
    __android_log_print (ANDROID_LOG_WARN, TAG, __VA_ARGS__)

#define nns_loge(...) \
    __android_log_print (ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define nns_logd(...) \
    __android_log_print (ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

#if GLIB_SIZEOF_VOID_P == 8
#define CAST_TO_LONG(p) (jlong)(p)
#define CAST_TO_TYPE(l,type) (type)(l)
#else
#define CAST_TO_LONG(p) (jlong)(jint)(p)
#define CAST_TO_TYPE(l,type) (type)(jint)(l)
#endif

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
  NNS_ELEMENT_TYPE_SWITCH_IN,
  NNS_ELEMENT_TYPE_SWITCH_OUT,

  NNS_ELEMENT_TYPE_UNKNOWN
} nns_element_type_e;

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
  jclass cls_tensors_data;
  jclass cls_tensors_info;
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
 * @brief Get element handle of given name.
 */
extern gpointer
nns_get_element_handle (pipeline_info_s * pipe_info, const gchar * name);

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
 * @brief Convert tensors data to TensorsData object.
 */
extern gboolean
nns_convert_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env, ml_tensors_data_h data_h, ml_tensors_info_h info_h, jobject * result);

/**
 * @brief Parse tensors data from TensorsData object.
 */
extern gboolean
nns_parse_tensors_data (pipeline_info_s * pipe_info, JNIEnv * env, jobject obj_data, ml_tensors_data_h * data_h, ml_tensors_info_h * info_h);

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

#endif /* __NNSTREAMER_ANDROID_NATIVE_H__ */
