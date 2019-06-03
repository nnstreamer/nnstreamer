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
#define NNS_TENSOR_RANK_LIMIT  (4)

/**
 * @brief The maximum number of other/tensor instances that other/tensors may have.
 * @since_tizen 5.5
 */
#define NNS_TENSOR_SIZE_LIMIT  (16)

/**
 * @brief Dimension information that NNStreamer support.
 * @since_tizen 5.5
 */
typedef unsigned int tensor_dim[NNS_TENSOR_RANK_LIMIT];

/**
 * @brief A handle of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *nns_pipeline_h;

/**
 * @brief A handle of a "sink node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *nns_sink_h;

/**
 * @brief A handle of a "src node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *nns_src_h;

/**
 * @brief A handle of a "switch" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *nns_switch_h;

/**
 * @brief A handle of a "valve node" of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef void *nns_valve_h;

/**
 * @brief Possible data element types of Tensor in NNStreamer.
 * @since_tizen 5.5
 */
typedef enum _nns_tensor_type_e
{
  NNS_TENSOR_TYPE_INT32 = 0,      /**< Integer 32bit */
  NNS_TENSOR_TYPE_UINT32,         /**< Unsigned integer 32bit */
  NNS_TENSOR_TYPE_INT16,          /**< Integer 16bit */
  NNS_TENSOR_TYPE_UINT16,         /**< Unsigned integer 16bit */
  NNS_TENSOR_TYPE_INT8,           /**< Integer 8bit */
  NNS_TENSOR_TYPE_UINT8,          /**< Unsigned integer 8bit */
  NNS_TENSOR_TYPE_FLOAT64,        /**< Float 64bit */
  NNS_TENSOR_TYPE_FLOAT32,        /**< Float 32bit */
  NNS_TENSOR_TYPE_INT64,          /**< Integer 64bit */
  NNS_TENSOR_TYPE_UINT64,         /**< Unsigned integer 64bit */
} nns_tensor_type_e;

/**
 * @brief Enumeration for the error codes of NNStreamer Pipelines.
 * @since_tizen 5.5
 */
typedef enum {
  NNS_ERROR_NONE				= TIZEN_ERROR_NONE, /**< Success! */
  NNS_ERROR_INVALID_PARAMETER			= TIZEN_ERROR_INVALID_PARAMETER, /**< Invalid parameter */
  NNS_ERROR_NOT_SUPPORTED			= TIZEN_ERROR_NOT_SUPPORTED, /**< The feature is not supported */
  NNS_ERROR_STREAMS_PIPE			= TIZEN_ERROR_STREAMS_PIPE, /**< Cannot create or access GStreamer pipeline. */
  NNS_ERROR_TRY_AGAIN				= TIZEN_ERROR_TRY_AGAIN, /**< The pipeline is not ready, yet (not negotiated, yet) */
} nns_error_e;

/**
 * @brief Enumeration for buffer deallocation policies.
 * @since_tizen 5.5
 */
typedef enum {
  NNS_BUF_FREE_BY_NNSTREAMER,	/**< Default. Application should not deallocate this buffer. NNStreamer will deallocate when the buffer is no more needed */
  NNS_BUF_DO_NOT_FREE1,		/**< This buffer is not to be freed by NNStreamer (i.e., it's a static object). However, be careful: NNStreamer might be accessing this object after the return of the API call. */
  NNS_BUF_POLICY_MAX,   /**< Max size of nns_buf_policy_e structure */
} nns_buf_policy_e;

/**
 * @brief Enumeration for nns pipeline state.
 * @since_tizen 5.5
 * @detail Refer to https://gstreamer.freedesktop.org/documentation/design/states.html.
 *         The state diagram of pipeline looks like this, assuming that there are no errors.
 *
 *          [ UNKNOWN ] "new null object"
 *               | "nns_pipeline_construct" starts
 *               V
 *          [  NULL   ] <------------------------------------------+
 *               | "nns_pipeline_construct" creates                |
 *               V                                                 |
 *          [  READY  ]                                            |
 *               | "nns_pipeline_construct' completes              | "nns_pipeline_destroy"
 *               V                                                 |
 *          [         ] ------------------------------------------>|
 *          [  PAUSED ] <-------------------+                      |
 *               | "nns_pipeline_start"     | "nns_pipeline_stop"  |
 *               V                          |                      |
 *          [ PLAYING ] --------------------+----------------------+
 *
 */
typedef enum {
  NNS_PIPELINE_STATE_UNKNOWN				= 0, /**< Unknown state. Maybe not constructed? */
  NNS_PIPELINE_STATE_NULL				= 1, /**< GST-State "Null" */
  NNS_PIPELINE_STATE_READY				= 2, /**< GST-State "Ready" */
  NNS_PIPELINE_STATE_PAUSED				= 3, /**< GST-State "Paused" */
  NNS_PIPELINE_STATE_PLAYING				= 4, /**< GST-State "Playing" */
} nns_pipeline_state_e;

/**
 * @brief Enumeration for switch types
 * @detail This designates different GStreamer filters, "GstInputSelector"/"GetOutputSelector".
 * @since_tizen 5.5
 */
typedef enum {
  NNS_SWITCH_OUTPUT_SELECTOR			= 0, /**< GstOutputSelector */
  NNS_SWITCH_INPUT_SELECTOR			= 1, /**< GstInputSelector */
} nns_switch_type_e;

/**
 * @brief Data structure for Tensor Information.
 * @since_tizen 5.5
 */
typedef struct {
  char * name;              /**< Name of each element in the tensor. */
  nns_tensor_type_e type;   /**< Type of each element in the tensor. */
  tensor_dim dimension;     /**< Dimension information. */
} nns_tensor_info_s;

/**
 * @brief Data structure for Tensors Information, which contains multiple tensors.
 * @since_tizen 5.5
 */
typedef struct {
  unsigned int num_tensors; /**< The number of tensors */
  nns_tensor_info_s info[NNS_TENSOR_SIZE_LIMIT];  /**< The list of tensor info */
} nns_tensors_info_s;

/**
 * @brief Callback for sink (tensor_sink) of nnstreamer pipelines (nnstreamer's output)
 * @detail If an application wants to accept data outputs of an nnstreamer stream, use this callback to get data from the stream. Note that the buffer may be deallocated after the return and this is synchronously called. Thus, if you need the data afterwards, copy the data to another buffer and return fast. Do not hold too much time in the callback. It is recommended to use very small tensors at sinks.
 * @since_tizen 5.5
 * @param[in] buf The contents of the tensor output (a single frame. tensor/tensors). Number of buf is determined by tensorsinfo->num_tensors.
 * @param[in] size The size of the buffer. Number of size is determined by tensorsinfo->num_tensors. Note that max num_tensors is 16 (NNS_TENSOR_SIZE_LIMIT).
 * @param[in] tensors_info The cardinality, dimension, and type of given tensor/tensors.
 * @param[in,out] pdata User Application's Private Data
 */
typedef void (*nns_sink_cb)
(const char *buf[], const size_t size[], const nns_tensors_info_s *tensors_info, void *pdata);

/****************************************************
 ** NNStreamer Pipeline Construction (gst-parse)   **
 ****************************************************/
/**
 * @brief Constructs the pipeline (GStreamer + NNStreamer)
 * @detail Uses this function to create a gst_parse_launch compatible NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a pipe handle must be released using nns_pipeline_destroy().
 * @param[in] pipeline_description The pipeline description compatible with GStreamer gst_parse_launch(). Refer to GStreamer manual or NNStreamer (github.com/nnsuite/nnstreamer) documentation for examples and the grammar.
 * @param[out] pipe The nnstreamer pipeline handler from the given description
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_STREAMS_PIPE Pipeline construction is failed because of wrong parameter or initialization failure.
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 */
int nns_pipeline_construct (const char *pipeline_description, nns_pipeline_h *pipe);

/**
 * @brief Destroys the pipeline
 * @detail Uses this function to destroy the pipeline constructed with nns_construct_pipeline.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be destroyed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_STREAMS_PIPE Cannot access the pipeline status.
 * @retval #NNS_ERROR_INVALID_PARAMETER The parameter is invalid (pipe is NULL?)
 */
int nns_pipeline_destroy (nns_pipeline_h pipe);

/**
 * @brief Gets the state of pipeline
 * @detail Gets the state of the pipeline handle returned by nns_construct_pipeline (pipe).
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be monitored.
 * @param[out] state The pipeline state.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 * @retval #NNS_ERROR_STREAMS_PIPE Failed to get state from the pipeline.
 */
int nns_pipeline_get_state (nns_pipeline_h pipe, nns_pipeline_state_e *state);

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
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_STREAMS_PIPE Failed to start.
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 */
int nns_pipeline_start (nns_pipeline_h pipe);

/**
 * @brief Stops the pipeline
 * @detail The pipeline handle returned by nns_construct_pipeline (pipe) is stopped.
 *         Note that this is asynchronous function. State might be "pending".
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be stopped.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_STREAMS_PIPE Failed to start.
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL?)
 */
int nns_pipeline_stop (nns_pipeline_h pipe);

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Registers a callback for sink (tensor_sink) of nnstreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a h handle must be unregistered using nns_pipeline_sink_unregister.
 * @param[in] pipe The pipeline to be attached with a sink node.
 * @param[in] sink_name The name of sink node, described with nns_pipeline_construct().
 * @param[in] cb The function to be called by the sink node.
 * @param[out] h The sink handle.
 * @param[in] pdata Private data for the callback. This value is passed to the callback when it's invoked.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_STREAMS_PIPE Failed to connect a signal to sink element.
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid. (pipe is NULL, sink_name is not found, or sink_name has an invalid type.)
 */
int nns_pipeline_sink_register (nns_pipeline_h pipe, const char *sink_name, nns_sink_cb cb, nns_sink_h *h, void *pdata);

/**
 * @brief Unregisters a callback for sink (tensor_sink) of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] The nns_sink_handle to be unregistered (destroyed)
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_sink_unregister (nns_sink_h h);

/**
 * @brief Gets a handle to operate as a src node of nnstreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a h handle must be released using nns_pipeline_src_put_handle().
 * @param[in] pipe The pipeline to be attached with a src node.
 * @param[in] src_name The name of src node, described with nns_pipeline_construct().
 * @param[out] tensors_info The cardinality, dimension, and type of given tensor/tensors.
 * @param[out] h The src handle.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #NNS_ERROR_STREAMS_PIPE Fail to get SRC element.
 */
int nns_pipeline_src_get_handle (nns_pipeline_h pipe, const char *src_name, nns_tensors_info_s *tensors_info, nns_src_h *h);

/**
 * @brief Closes the given handle of a src node of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] h The src handle to be put (closed).
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_src_put_handle (nns_src_h h);

/**
 * @brief Puts an input data frame.
 * @param[in] h The nns_src_handle returned by nns_pipeline_gethandle().
 * @param[in] policy The policy of buf deallocation.
 * @param[in] buf The input buffers, in the format of tensorsinfo given by nns_pipeline_gethandle()
 * @param[in] size The sizes of input buffers. This must be consistent with the given tensorsinfo, probed by nns_pipeline_src_get_handle().
 * @param[in] num_tensors The number of tensors (number of buf and number of size) for the input frame. This must be consistent with the given tensorinfo, probed by nns_pipeline_src_get_handle(). MAX is 16 (NNS_TENSOR_SIZE_LIMIT).
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #NNS_ERROR_STREAMS_PIPE The pipeline has inconsistent padcaps. Not negotiated?
 * @retval #NNS_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int nns_pipeline_src_input_data (nns_src_h h, nns_buf_policy_e policy, char *buf[], const size_t size[], unsigned int num_tensors);

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Gets a handle to operate a "GstInputSelector / GstOutputSelector" node of nnstreamer pipelines.
 * @detail Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-bad-plugins/html/gst-plugins-bad-plugins-input-selector.html for input selectors.
 *         Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-output-selector.html for output selectors.
 * @remarks If the function succeeds, @a h handle must be released using nns_pipeline_switch_put_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] switch_name The name of switch (InputSelector/OutputSelector)
 * @param[out] type The type of the switch. If NULL, it is ignored.
 * @param[out] h The switch handle.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_switch_get_handle (nns_pipeline_h pipe, const char *switch_name, nns_switch_type_e *type, nns_switch_h *h);

/**
 * @brief Closes the given switch handle.
 * @param[in] h The handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_switch_put_handle (nns_switch_h h);

/**
 * @brief Controls the switch with the given handle to select input/output nodes(pads).
 * @param[in] h The switch handle returned by nns_pipeline_switch_get_handle()
 * @param[in] pad_name The name of the chosen pad to be activated. Use nns_pipeline_switch_nodelist to list the available pad names.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_switch_select (nns_switch_h h, const char *pad_name);

/**
 * @brief Gets the pad names of a switch.
 * @param[in] h The switch handle returned by nns_pipeline_switch_get_handle()
 * @param[out] list NULL terminated array of char*. The caller must free each string (char*) in the list and free the list itself.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #NNS_ERROR_STREAMS_PIPE The element is not both input and output switch (Internal data inconsistency).
 */
int nns_pipeline_switch_nodelist (nns_switch_h h, char *** list);

/**
 * @brief Gets a handle to operate a "GstValve" node of nnstreamer pipelines.
 * @detail Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-valve.html for more info.
 * @remarks If the function succeeds, @a h handle must be released using nns_pipeline_valve_put_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] valve_name The name of valve (Valve)
 * @param[out] h The valve handle.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_valve_get_handle (nns_pipeline_h pipe, const char *valve_name, nns_valve_h *h);

/**
 * @brief Closes the given valve handle.
 * @param[in] h The handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_valve_put_handle (nns_valve_h h);

/**
 * @brief Controls the valve with the given handle.
 * @param[in] h The valve handle returned by nns_pipeline_valve_get_handle()
 * @param[in] valve_drop 1 to close (drop & stop the flow). 0 to open (let the flow pass)
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 * @retval #NNS_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int nns_pipeline_valve_control (nns_valve_h h, int valve_drop);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNSTREAMER_H__ */
