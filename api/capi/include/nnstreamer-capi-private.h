/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_CAPI_PRIVATE_H__
#define __NNSTREAMER_CAPI_PRIVATE_H__

#include <glib.h>
#include <gst/gst.h>

#include "nnstreamer.h"
#include "nnstreamer-single.h"
#include "tensor_typedef.h"
#include "nnstreamer_log.h"

/* Tizen ML feature */
#if defined (__TIZEN__)
#include "nnstreamer-tizen-internal.h"

typedef enum
{
  NOT_CHECKED_YET = -1,
  NOT_SUPPORTED = 0,
  SUPPORTED = 1
} feature_state_t;

#if defined (__FEATURE_CHECK_SUPPORT__)
#define check_feature_state() \
  do { \
    int feature_ret = ml_tizen_get_feature_enabled (); \
    if (ML_ERROR_NONE != feature_ret) \
      return feature_ret; \
  } while (0);

#define set_feature_state(...) ml_tizen_set_feature_state(__VA_ARGS__)
#else
#define check_feature_state()
#define set_feature_state(...)
#endif  /* __FEATURE_CHECK_SUPPORT__ */

#if defined (__PRIVILEGE_CHECK_SUPPORT__)

#define convert_tizen_element(...) ml_tizen_convert_element(__VA_ARGS__)

#if (TIZENVERSION >= 5) && (TIZENVERSION < 9999)
#define get_tizen_resource(...) ml_tizen_get_resource(__VA_ARGS__)
#define release_tizen_resource(...) ml_tizen_release_resource(__VA_ARGS__)
#define TIZEN5PLUS 1

#elif (TIZENVERSION < 5)
#define get_tizen_resource(...) (0)
#define release_tizen_resource(...) do { } while (0)
typedef void * mm_resource_manager_h;
typedef enum { MM_RESOURCE_MANAGER_RES_TYPE_MAX } mm_resource_manager_res_type_e;
#define TIZEN5PLUS 0

#else
#error Tizen version is not defined.
#endif

#else

#define convert_tizen_element(...) ML_ERROR_NONE
#define get_tizen_resource(...) ML_ERROR_NONE
#define release_tizen_resource(...)
#define TIZEN5PLUS 0

#endif  /* __PRIVILEGE_CHECK_SUPPORT__ */

#else
#define check_feature_state()
#define set_feature_state(...)
#define convert_tizen_element(...) ML_ERROR_NONE
#define get_tizen_resource(...) ML_ERROR_NONE
#define release_tizen_resource(...)
#define TIZEN5PLUS 0
#endif  /* __TIZEN__ */

#define EOS_MESSAGE_TIME_LIMIT 100
#define WAIT_PAUSED_TIME_LIMIT 100

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Internal private representation of custom filter handle.
 */
typedef struct {
  char *name;
  ml_tensors_info_h in_info;
  ml_tensors_info_h out_info;
  ml_custom_easy_invoke_cb cb;
  void *pdata;
} ml_custom_filter_s;

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
  void *handle; /**< The handle which owns this buffer and will be used to de-alloc the data */
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
  ML_PIPELINE_ELEMENT_COMMON = 0xB,
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
  GstElement *element;            /**< The pipeline itself (GstPipeline) */
  GstBus *bus;                    /**< The bus of the pipeline */
  gulong signal_msg;              /**< The message signal (connected to bus) */
  GMutex lock;                    /**< Lock for pipeline operations */
  gboolean isEOS;                 /**< The pipeline is EOS state */
  ml_pipeline_state_e pipe_state; /**< The state of pipeline */
  GHashTable *namednodes;         /**< hash table of "element"s. */
  GHashTable *resources;          /**< hash table of resources to construct the pipeline */
  pipeline_state_cb_s state_cb;   /**< Callback to notify the change of pipeline state */
};

/**
 * @brief Internal private representation sink callback function for GstTensorSink and GstAppSink
 * @details This represents a single instance of callback registration. This should not be exposed to applications.
 */
typedef struct {
  ml_pipeline_sink_cb cb;
  void *pdata;
} callback_info_s;

/**
 * @brief Internal private representation of common element handle (All GstElement except AppSink and TensorSink)
 * @details This represents a single instance of registration. This should not be exposed to applications.
 */
typedef struct _ml_pipeline_common_elem {
  ml_pipeline *pipe;
  ml_pipeline_element *element;
  guint32 id;
  callback_info_s *callback_info;   /**< Callback function information. If element is not GstTensorSink or GstAppSink, then it should be NULL. */
} ml_pipeline_common_elem;

/**
 * @brief An information to create single-shot instance.
 */
typedef struct {
  ml_tensors_info_h input_info;  /**< The input tensors information. */
  ml_tensors_info_h output_info; /**< The output tensors information. */
  ml_nnfw_type_e nnfw;           /**< The neural network framework. */
  ml_nnfw_hw_e hw;               /**< The type of hardware resource. */
  char *models;                  /**< Comma separated neural network model files. */
  char *custom_option;           /**< Custom option string for neural network framework. */
} ml_single_preset;

/**
 * @brief Opens an ML model with the custom options and returns the instance as a handle.
 * This is internal function to handle various options in public APIs.
 */
int ml_single_open_custom (ml_single_h *single, ml_single_preset *info);

/**
 * @brief Macro to check the availability of given NNFW.
 */
#define ml_nnfw_is_available(f,h) ({bool a; (ml_check_nnfw_availability ((f), (h), &a) == ML_ERROR_NONE && a);})

/**
 * @brief Macro to check the tensors info is valid.
 */
#define ml_tensors_info_is_valid(i) ({bool v; (ml_tensors_info_validate ((i), &v) == ML_ERROR_NONE && v);})

/**
 * @brief Macro to compare the tensors info.
 */
#define ml_tensors_info_is_equal(i1,i2) ({bool e; (ml_tensors_info_compare ((i1), (i2), &e) == ML_ERROR_NONE && e);})

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
 * @brief Compares the given tensors information.
 * @details If the function returns an error, @a equal is not changed.
 * @since_tizen 6.0
 * @param[in] info1 The handle of tensors information to be compared.
 * @param[in] info2 The handle of tensors information to be compared.
 * @param[out] equal @c true if given tensors information is equal, @c false if it's not equal.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_compare (const ml_tensors_info_h info1, const ml_tensors_info_h info2, bool *equal);

/**
 * @brief Frees the tensors info pointer.
 * @since_tizen 5.5
 * @param[in] info The tensors info pointer to be freed.
 */
void ml_tensors_info_free (ml_tensors_info_s *info);

/**
 * @brief Allocates a tensors information handle from gst info.
 */
int ml_tensors_info_create_from_gst (ml_tensors_info_h *ml_info, GstTensorsInfo *gst_info);

/**
 * @brief Copies tensor metadata from gst tensors info.
 */
void ml_tensors_info_copy_from_gst (ml_tensors_info_s *ml_info, const GstTensorsInfo *gst_info);

/**
 * @brief Copies tensor metadata from ml tensors info.
 */
void ml_tensors_info_copy_from_ml (GstTensorsInfo *gst_info, const ml_tensors_info_s *ml_info);

/**
 * @brief Creates a tensor data frame without buffer with the given tensors information.
 * @details If @a info is null, this allocates data handle with empty tensor data.
 * @param[in] info The handle of tensors information for the allocation.
 * @param[out] data The handle of tensors data.
 * @return @c 0 on success. Otherwise a negative error value.
 */
int ml_tensors_data_create_no_alloc (const ml_tensors_info_h info, ml_tensors_data_h *data);

/**
 * @brief Creates a tensor data frame without allocating new buffer cloning the given tensors data.
 * @details If @a data_src is null, this returns error.
 * @param[in] data_src The handle of tensors data to be cloned.
 * @param[out] data The handle of tensors data.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_tensors_data_clone_no_alloc (const ml_tensors_data_s * data_src, ml_tensors_data_h * data);

/**
 * @brief Initializes the GStreamer library. This is internal function.
 */
int ml_initialize_gstreamer (void);

/**
 * @brief Validates the nnfw model file.
 * @since_tizen 5.5
 * @param[in] model List of model file paths.
 * @param[in] num_models The number of model files. There are a few frameworks that require multiple model files for a single model.
 * @param[in/out] nnfw The type of NNFW.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_validate_model_file (char **model, unsigned int num_models, ml_nnfw_type_e * nnfw);

/**
 * @brief Checks the availability of the plugin.
 */
int ml_check_plugin_availability (const char *plugin_name, const char *element_name);

/**
 * @brief Internal function to convert accelerator as tensor_filter property format.
 * @note returned value must be freed by the caller
 */
char* ml_nnfw_to_str_prop (ml_nnfw_hw_e hw);

/**
 * @brief Internal function to get the sub-plugin name.
 */
const char* ml_get_nnfw_subplugin_name (ml_nnfw_type_e nnfw);

/**
 * @brief Gets the element of pipeline itself (GstElement).
 * @details With the returned reference, you can use GStreamer functions to handle the element in pipeline.
 *          Note that caller should release the returned reference using gst_object_unref().
 * @return The reference of pipeline itself. Null if the pipeline is not constructed or closed.
 */
GstElement* ml_pipeline_get_gst_element (ml_pipeline_h pipe);

int ml_single_destroy_notify (ml_single_h single, ml_tensors_data_s *data);

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
