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
 * @file nnstreamer.h
 * @date 07 March 2019
 * @brief Tizen NNStreamer/Pipeline(main) C-API Header.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNSTREAMER_H__
#define __TIZEN_MACHINELEARNING_NNSTREAMER_H__

#include <stddef.h>
#include <tizen_error.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNSTREAMER_PIPELINE_MODULE
 * @{
 */

/**
 * @brief The maximum rank that NNStreamer supports with Tizen APIs.
 * @since_tizen 5.5
 */
#define ML_TENSOR_RANK_LIMIT  (4)

/**
 * @brief The maximum number of other/tensor instances that other/tensors may have.
 * @since_tizen 5.5
 */
#define ML_TENSOR_SIZE_LIMIT  (16)

/**
 * @brief Dimension information that NNStreamer support.
 * @since_tizen 5.5
 */
typedef unsigned int ml_tensor_dim[ML_TENSOR_RANK_LIMIT];

/**
 * @brief A handle of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_h;

/**
 * @brief A handle of a "sink node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_sink_h;

/**
 * @brief A handle of a "src node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_src_h;

/**
 * @brief A handle of a "switch" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_switch_h;

/**
 * @brief A handle of a "valve node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *ml_pipeline_valve_h;

/**
 * @brief Possible data element types of Tensor in NNStreamer.
 * @since_tizen 5.5
 */
typedef enum _ml_tensor_type_e
{
  ML_TENSOR_TYPE_INT32 = 0,      /**< Integer 32bit */
  ML_TENSOR_TYPE_UINT32,         /**< Unsigned integer 32bit */
  ML_TENSOR_TYPE_INT16,          /**< Integer 16bit */
  ML_TENSOR_TYPE_UINT16,         /**< Unsigned integer 16bit */
  ML_TENSOR_TYPE_INT8,           /**< Integer 8bit */
  ML_TENSOR_TYPE_UINT8,          /**< Unsigned integer 8bit */
  ML_TENSOR_TYPE_FLOAT64,        /**< Float 64bit */
  ML_TENSOR_TYPE_FLOAT32,        /**< Float 32bit */
  ML_TENSOR_TYPE_INT64,          /**< Integer 64bit */
  ML_TENSOR_TYPE_UINT64,         /**< Unsigned integer 64bit */
  ML_TENSOR_TYPE_UNKNOWN          /**< Unknown type */
} ml_tensor_type_e;

/**
 * @brief Enumeration for the error codes of NNStreamer Pipelines.
 * @since_tizen 5.5
 */
typedef enum {
  ML_ERROR_NONE				= TIZEN_ERROR_NONE, /**< Success! */
  ML_ERROR_INVALID_PARAMETER			= TIZEN_ERROR_INVALID_PARAMETER, /**< Invalid parameter */
  ML_ERROR_NOT_SUPPORTED			= TIZEN_ERROR_NOT_SUPPORTED, /**< The feature is not supported */
  ML_ERROR_STREAMS_PIPE			= TIZEN_ERROR_STREAMS_PIPE, /**< Cannot create or access GStreamer pipeline. */
  ML_ERROR_TRY_AGAIN				= TIZEN_ERROR_TRY_AGAIN, /**< The pipeline is not ready, yet (not negotiated, yet) */
} ml_error_e;

/**
 * @brief Enumeration for buffer deallocation policies.
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_BUF_POLICY_AUTO_FREE,	/**< Default. Application should not deallocate this buffer. NNStreamer will deallocate when the buffer is no more needed */
  ML_PIPELINE_BUF_POLICY_DO_NOT_FREE,		/**< This buffer is not to be freed by NNStreamer (i.e., it's a static object). However, be careful: NNStreamer might be accessing this object after the return of the API call. */
  ML_PIPELINE_BUF_POLICY_MAX,   /**< Max size of ml_pipeline_buf_policy_e structure */
} ml_pipeline_buf_policy_e;

/**
 * @brief Enumeration for pipeline state.
 * @since_tizen 5.5
 * @detail Refer to https://gstreamer.freedesktop.org/documentation/plugin-development/basics/states.html.
 *         The state diagram of pipeline looks like this, assuming that there are no errors.
 *
 *          [ UNKNOWN ] "new null object"
 *               | "ml_pipeline_construct" starts
 *               V
 *          [  NULL   ] <------------------------------------------+
 *               | "ml_pipeline_construct" creates                |
 *               V                                                 |
 *          [  READY  ]                                            |
 *               | "ml_pipeline_construct' completes              | "ml_pipeline_destroy"
 *               V                                                 |
 *          [         ] ------------------------------------------>|
 *          [  PAUSED ] <-------------------+                      |
 *               | "ml_pipeline_start"     | "ml_pipeline_stop"  |
 *               V                          |                      |
 *          [ PLAYING ] --------------------+----------------------+
 *
 */
typedef enum {
  ML_PIPELINE_STATE_UNKNOWN				= 0, /**< Unknown state. Maybe not constructed? */
  ML_PIPELINE_STATE_NULL				= 1, /**< GST-State "Null" */
  ML_PIPELINE_STATE_READY				= 2, /**< GST-State "Ready" */
  ML_PIPELINE_STATE_PAUSED				= 3, /**< GST-State "Paused" */
  ML_PIPELINE_STATE_PLAYING				= 4, /**< GST-State "Playing" */
} ml_pipeline_state_e;

/**
 * @brief Enumeration for switch types
 * @detail This designates different GStreamer filters, "GstInputSelector"/"GetOutputSelector".
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_SWITCH_OUTPUT_SELECTOR			= 0, /**< GstOutputSelector */
  ML_PIPELINE_SWITCH_INPUT_SELECTOR			= 1, /**< GstInputSelector */
} ml_pipeline_switch_type_e;

/**
 * @brief Data structure for Tensor Information.
 * @since_tizen 5.5
 */
typedef struct {
  char * name;              /**< Name of each element in the tensor. */
  ml_tensor_type_e type;   /**< Type of each element in the tensor. */
  ml_tensor_dim dimension;     /**< Dimension information. */
} ml_tensor_info_s;

/**
 * @brief Data structure for Tensors Information, which contains multiple tensors.
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
 * @brief An instance of input or output frames. ml_tensors_info_s is the metadata.
 * @since_tizen 5.5
 */
typedef struct {
  unsigned int num_tensors; /**< The number of tensors. */
  ml_tensor_data_s tensors[ML_TENSOR_SIZE_LIMIT]; /**< The list of tensor data. NULL for unused tensors. */
} ml_tensors_data_s;

/**
 * @brief Callback for sink element of nnstreamer pipelines (pipeline's output)
 * @detail If an application wants to accept data outputs of an nnstreamer stream, use this callback to get data from the stream. Note that the buffer may be deallocated after the return and this is synchronously called. Thus, if you need the data afterwards, copy the data to another buffer and return fast. Do not hold too much time in the callback. It is recommended to use very small tensors at sinks.
 * @since_tizen 5.5
 * @param[in] data The contents of the tensor output (a single frame. tensor/tensors). Number of tensors is determined by data->num_tensors. Note that max num_tensors is 16 (ML_TENSOR_SIZE_LIMIT).
 * @param[in] info The cardinality, dimension, and type of given tensor/tensors.
 * @param[in,out] pdata User Application's Private Data.
 */
typedef void (*ml_pipeline_sink_cb) (const ml_tensors_data_s *data, const ml_tensors_info_s *info, void *pdata);

/****************************************************
 ** NNStreamer Pipeline Construction (gst-parse)   **
 ****************************************************/
/**
 * @brief Constructs the pipeline (GStreamer + NNStreamer)
 * @detail Uses this function to create a gst_parse_launch compatible NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a pipe handle must be released using ml_pipeline_destroy().
 * @param[in] pipeline_description The pipeline description compatible with GStreamer gst_parse_launch(). Refer to GStreamer manual or NNStreamer (github.com/nnsuite/nnstreamer) documentation for examples and the grammar.
 * @param[out] pipe The nnstreamer pipeline handler from the given description
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_STREAMS_PIPE Pipeline construction is failed because of wrong parameter or initialization failure.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 */
int ml_pipeline_construct (const char *pipeline_description, ml_pipeline_h *pipe);

/**
 * @brief Destroys the pipeline
 * @detail Uses this function to destroy the pipeline constructed with nns_construct_pipeline.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be destroyed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_STREAMS_PIPE Cannot access the pipeline status.
 * @retval #ML_ERROR_INVALID_PARAMETER The parameter is invalid (pipe is NULL?)
 */
int ml_pipeline_destroy (ml_pipeline_h pipe);

/**
 * @brief Gets the state of pipeline
 * @detail Gets the state of the pipeline handle returned by nns_construct_pipeline (pipe).
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be monitored.
 * @param[out] state The pipeline state.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to get state from the pipeline.
 */
int ml_pipeline_get_state (ml_pipeline_h pipe, ml_pipeline_state_e *state);

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Starts the pipeline
 * @detail The pipeline handle returned by nns_construct_pipeline (pipe) is started.
 *         Note that this is asynchronous function. State might be "pending".
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be started.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_STREAMS_PIPE Failed to start.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 */
int ml_pipeline_start (ml_pipeline_h pipe);

/**
 * @brief Stops the pipeline
 * @detail The pipeline handle returned by nns_construct_pipeline (pipe) is stopped.
 *         Note that this is asynchronous function. State might be "pending".
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be stopped.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_STREAMS_PIPE Failed to start.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 */
int ml_pipeline_stop (ml_pipeline_h pipe);

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Registers a callback for sink (tensor_sink) of nnstreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a h handle must be unregistered using ml_pipeline_sink_unregister.
 * @param[in] pipe The pipeline to be attached with a sink node.
 * @param[in] sink_name The name of sink node, described with ml_pipeline_construct().
 * @param[in] cb The function to be called by the sink node.
 * @param[out] h The sink handle.
 * @param[in] pdata Private data for the callback. This value is passed to the callback when it's invoked.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_STREAMS_PIPE Failed to connect a signal to sink element.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL, sink_name is not found, or sink_name has an invalid type.)
 */
int ml_pipeline_sink_register (ml_pipeline_h pipe, const char *sink_name, ml_pipeline_sink_cb cb, ml_pipeline_sink_h *h, void *pdata);

/**
 * @brief Unregisters a callback for sink (tensor_sink) of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] h The sink handle to be unregistered (destroyed)
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_sink_unregister (ml_pipeline_sink_h h);

/**
 * @brief Gets a handle to operate as a src node of nnstreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a h handle must be released using ml_pipeline_src_put_handle().
 * @param[in] pipe The pipeline to be attached with a src node.
 * @param[in] src_name The name of src node, described with ml_pipeline_construct().
 * @param[out] tensors_info The cardinality, dimension, and type of given tensor/tensors.
 * @param[out] h The src handle.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Fail to get SRC element.
 */
int ml_pipeline_src_get_handle (ml_pipeline_h pipe, const char *src_name, ml_tensors_info_s *tensors_info, ml_pipeline_src_h *h);

/**
 * @brief Closes the given handle of a src node of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] h The src handle to be put (closed).
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_src_put_handle (ml_pipeline_src_h h);

/**
 * @brief Puts an input data frame.
 * @param[in] h The source handle returned by ml_pipeline_src_get_handle().
 * @param[in] data The input tensors, in the format of tensors info given by ml_pipeline_src_get_handle().
 * @param[in] policy The policy of buf deallocation.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The pipeline has inconsistent padcaps. Not negotiated?
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int ml_pipeline_src_input_data (ml_pipeline_src_h h, const ml_tensors_data_s *data, ml_pipeline_buf_policy_e policy);

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Gets a handle to operate a "GstInputSelector / GstOutputSelector" node of nnstreamer pipelines.
 * @detail Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-input-selector.html for input selectors.
 *         Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-output-selector.html for output selectors.
 * @remarks If the function succeeds, @a h handle must be released using ml_pipeline_switch_put_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] switch_name The name of switch (InputSelector/OutputSelector)
 * @param[out] type The type of the switch. If NULL, it is ignored.
 * @param[out] h The switch handle.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_get_handle (ml_pipeline_h pipe, const char *switch_name, ml_pipeline_switch_type_e *type, ml_pipeline_switch_h *h);

/**
 * @brief Closes the given switch handle.
 * @param[in] h The handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_put_handle (ml_pipeline_switch_h h);

/**
 * @brief Controls the switch with the given handle to select input/output nodes(pads).
 * @param[in] h The switch handle returned by ml_pipeline_switch_get_handle()
 * @param[in] pad_name The name of the chosen pad to be activated. Use ml_pipeline_switch_nodelist to list the available pad names.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_select (ml_pipeline_switch_h h, const char *pad_name);

/**
 * @brief Gets the pad names of a switch.
 * @param[in] h The switch handle returned by ml_pipeline_switch_get_handle()
 * @param[out] list NULL terminated array of char*. The caller must free each string (char*) in the list and free the list itself.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The element is not both input and output switch (Internal data inconsistency).
 */
int ml_pipeline_switch_nodelist (ml_pipeline_switch_h h, char *** list);

/**
 * @brief Gets a handle to operate a "GstValve" node of nnstreamer pipelines.
 * @detail Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-valve.html for more info.
 * @remarks If the function succeeds, @a h handle must be released using ml_pipeline_valve_put_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] valve_name The name of valve (Valve)
 * @param[out] h The valve handle.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_valve_get_handle (ml_pipeline_h pipe, const char *valve_name, ml_pipeline_valve_h *h);

/**
 * @brief Closes the given valve handle.
 * @param[in] h The handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_valve_put_handle (ml_pipeline_valve_h h);

/**
 * @brief Controls the valve with the given handle.
 * @param[in] h The valve handle returned by ml_pipeline_valve_get_handle()
 * @param[in] drop 1 to close (drop & stop the flow). 0 to open (let the flow pass)
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_valve_control (ml_pipeline_valve_h h, int drop);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNSTREAMER_H__ */
