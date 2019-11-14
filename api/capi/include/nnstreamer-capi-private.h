/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */
/**
 * @file nnstreamer-capi-private.h
 * @date 07 March 2019
 * @brief NNStreamer/Pipeline(main) C-API Private Header.
 *        This file should NOT be exported to SDK or devel package.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_CAPI_PRIVATE_H__
#define __NNSTREAMER_CAPI_PRIVATE_H__

#include <glib.h>
#include <gst/gst.h>

#include "nnstreamer.h"
#include "tensor_typedef.h"

/* Tizen ML feature */
#if defined (__TIZEN__)
#include "nnstreamer-tizen-internal.h"

#define check_feature_state() \
  do { \
    int feature_ret = ml_tizen_get_feature_enabled (); \
    if (ML_ERROR_NONE != feature_ret) \
      return feature_ret; \
  } while (0);

#define set_feature_state(...) ml_tizen_set_feature_state(__VA_ARGS__)
#define convert_tizen_element(...) ml_tizen_convert_element(__VA_ARGS__)
#define get_tizen_resource(...) ml_tizen_get_resource(__VA_ARGS__)
#define release_tizen_resource(...) ml_tizen_release_resource(__VA_ARGS__)
#else
#define check_feature_state()
#define set_feature_state(...)
#define convert_tizen_element(...) ML_ERROR_NONE
#define get_tizen_resource(...) ML_ERROR_NONE
#define release_tizen_resource(...)
#endif

#define TAG_NAME "nnstreamer-capi"

#if defined(__TIZEN__)
  #include <dlog.h>

  #define ml_logi(...) \
      dlog_print (DLOG_INFO, TAG_NAME, __VA_ARGS__)

  #define ml_logw(...) \
      dlog_print (DLOG_WARN, TAG_NAME, __VA_ARGS__)

  #define ml_loge(...) \
      dlog_print (DLOG_ERROR, TAG_NAME, __VA_ARGS__)

  #define ml_logd(...) \
      dlog_print (DLOG_DEBUG, TAG_NAME, __VA_ARGS__)

#elif defined(__ANDROID__)
  #include <android/log.h>

  #define ml_logi(...) \
      __android_log_print (ANDROID_LOG_INFO, TAG_NAME, __VA_ARGS__)

  #define ml_logw(...) \
      __android_log_print (ANDROID_LOG_WARN, TAG_NAME, __VA_ARGS__)

  #define ml_loge(...) \
      __android_log_print (ANDROID_LOG_ERROR, TAG_NAME, __VA_ARGS__)

  #define ml_logd(...) \
      __android_log_print (ANDROID_LOG_DEBUG, TAG_NAME, __VA_ARGS__)

#else /* Linux distro */
  #define ml_logi g_info
  #define ml_logw g_warning
  #define ml_loge g_critical
  #define ml_logd g_debug
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Data structure for tensor information.
 * @since_tizen 5.5
 */
typedef struct {
  char *name;              /**< Name of each element in the tensor. */
  ml_tensor_type_e type;   /**< Type of each element in the tensor. */
  ml_tensor_dimension dimension;     /**< Dimension information. */
} ml_tensor_info_s;

/**
 * @brief Data structure for tensors information, which contains multiple tensors.
 * @since_tizen 5.5
 */
typedef struct {
  unsigned int num_tensors; /**< The number of tensors. */
  ml_tensor_info_s info[ML_TENSOR_SIZE_LIMIT];  /**< The list of tensor info. */
} ml_tensors_info_s;

/**
 * @brief An instance of a single input or output frame.
 * @since_tizen 5.5
 */
typedef struct {
  void *tensor; /**< The instance of tensor data. */
  size_t size; /**< The size of tensor. */
} ml_tensor_data_s;

/**
 * @brief An instance of input or output frames. #ml_tensors_info_h is the handle for tensors metadata.
 * @since_tizen 5.5
 */
typedef struct {
  unsigned int num_tensors; /**< The number of tensors. */
  ml_tensor_data_s tensors[ML_TENSOR_SIZE_LIMIT]; /**< The list of tensor data. NULL for unused tensors. */
} ml_tensors_data_s;

/**
 * @brief Possible controls on elements of a pipeline.
 */
typedef enum {
  ML_PIPELINE_ELEMENT_UNKNOWN = 0x0,
  ML_PIPELINE_ELEMENT_SINK = 0x1,
  ML_PIPELINE_ELEMENT_APP_SRC = 0x2,
  ML_PIPELINE_ELEMENT_APP_SINK = 0x3,
  ML_PIPELINE_ELEMENT_VALVE = 0x4,
  ML_PIPELINE_ELEMENT_SWITCH_INPUT = 0x8,
  ML_PIPELINE_ELEMENT_SWITCH_OUTPUT = 0x9,
} ml_pipeline_element_e;

/**
 * @brief Internal private representation of pipeline handle.
 */
typedef struct _ml_pipeline ml_pipeline;

/**
 * @brief An element that may be controlled individually in a pipeline.
 */
typedef struct _ml_pipeline_element {
  GstElement *element; /**< The Sink/Src/Valve/Switch element */
  ml_pipeline *pipe; /**< The main pipeline */
  char *name;
  ml_pipeline_element_e type;
  GstPad *src;
  GstPad *sink; /**< Unref this at destroy */
  ml_tensors_info_s tensors_info;
  size_t size;

  GList *handles;
  int maxid; /**< to allocate id for each handle */
  gulong handle_id;

  GMutex lock; /**< Lock for internal values */
  gboolean is_media_stream;
} ml_pipeline_element;

/**
 * @brief Internal data structure for the pipeline state callback.
 */
typedef struct {
  ml_pipeline_state_cb cb; /**< Callback to notify the change of pipeline state */
  void *user_data; /**< The user data passed when calling the state change callback */
} pipeline_state_cb_s;

/**
 * @brief Internal data structure for the resource.
 */
typedef struct {
  gchar *type; /**< resource type */
  gpointer handle; /**< pointer to resource handle */
} pipeline_resource_s;

/**
 * @brief Internal private representation of pipeline handle.
 * @details This should not be exposed to applications
 */
struct _ml_pipeline {
  GstElement *element;    /**< The pipeline itself (GstPipeline) */
  GstBus *bus;            /**< The bus of the pipeline */
  gulong signal_msg;      /**< The message signal (connected to bus) */
  GMutex lock;            /**< Lock for pipeline operations */
  GHashTable *namednodes; /**< hash table of "element"s. */
  GHashTable *resources;  /**< hash table of resources to construct the pipeline */
  pipeline_state_cb_s state_cb; /**< Callback to notify the change of pipeline state */
};

/**
 * @brief Internal private representation of sink handle of GstTensorSink and GstAppSink
 * @details This represents a single instance of callback registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_sink {
  ml_pipeline *pipe; /**< The pipeline, which is the owner of this ml_pipeline_sink */
  ml_pipeline_element *element;
  guint32 id;
  ml_pipeline_sink_cb cb;
  void *pdata;
} ml_pipeline_sink;

/**
 * @brief Internal private representation of src handle of GstAppSrc
 * @details This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_src {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
} ml_pipeline_src;

/**
 * @brief Internal private representation of switch handle (GstInputSelector, GstOutputSelector)
 * @details This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_switch {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
} ml_pipeline_switch;

/**
 * @brief Internal private representation of valve handle (GstValve)
 * @details This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_valve {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
} ml_pipeline_valve;

/**
 * @brief Macro to check the tensors info is valid.
 * @since_tizen 5.5
 */
#define ml_tensors_info_is_valid(i,v) (ml_tensors_info_validate ((i), &(v)) == ML_ERROR_NONE && (v))

/**
 * @brief Gets the byte size of the given tensor info.
 */
size_t ml_tensor_info_get_size (const ml_tensor_info_s *info);

/**
 * @brief Initializes the tensors information with default value.
 * @since_tizen 5.5
 * @param[in] info The tensors info pointer to be initialized.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_initialize (ml_tensors_info_s *info);

/**
 * @brief Frees the tensors info pointer.
 * @since_tizen 5.5
 * @param[in] info The tensors info pointer to be freed.
 */
void ml_tensors_info_free (ml_tensors_info_s *info);

/**
 * @brief Copies tensor metadata from gst tensors info.
 */
void ml_tensors_info_copy_from_gst (ml_tensors_info_s *ml_info, const GstTensorsInfo *gst_info);

/**
 * @brief Copies tensor metadata from ml tensors info.
 */
void ml_tensors_info_copy_from_ml (GstTensorsInfo *gst_info, const ml_tensors_info_s *ml_info);

/**
 * @brief Gets caps from tensors info.
 */
GstCaps * ml_tensors_info_get_caps (const ml_tensors_info_s *info);

/**
 * @brief Creates a tensor data frame wihout buffer with the given tensors information.
 * @param[in] info The handle of tensors information for the allocation.
 * @param[out] data The handle of tensors data.
 * @return @c 0 on success. Otherwise a negative error value.
 */
int ml_tensors_data_create_no_alloc (const ml_tensors_info_h info, ml_tensors_data_h *data);

/**
 * @brief Initializes the GStreamer library. This is internal function.
 */
int ml_initialize_gstreamer (void);

/**
 * @brief Validates the nnfw model file.
 * @since_tizen 5.5
 * @param[in] model The path of model file.
 * @param[in/out] nnfw The type of NNFW.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_validate_model_file (const char *model, ml_nnfw_type_e * nnfw);

/**
 * @brief Checks the availability of the plugin.
 */
int ml_check_plugin_availability (const char *plugin_name, const char *element_name);

#if defined (__TIZEN__)
/**
 * @brief Checks whether machine_learning.inference feature is enabled or not.
 */
int ml_tizen_get_feature_enabled (void);

/**
 * @brief Set the feature status of machine_learning.inference.
 * This is only used for Unit test.
 */
int ml_tizen_set_feature_state (int state);

/**
 * @brief Releases the resource handle of Tizen.
 */
void ml_tizen_release_resource (gpointer handle, const gchar * res_type);

/**
 * @brief Gets the resource handle of Tizen.
 */
int ml_tizen_get_resource (ml_pipeline_h pipe, const gchar * res_type);

/**
 * @brief Converts predefined element for Tizen.
 */
int ml_tizen_convert_element (ml_pipeline_h pipe, gchar ** result, gboolean is_internal);
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __NNSTREAMER_CAPI_PRIVATE_H__ */
