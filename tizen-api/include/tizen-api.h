/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file tizen-api.h
 * @date 07 March 2019
 * @brief Tizen NNStreamer/Pipeline(main) C-API Header.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_NNSTREAMER_API_MAIN_H__
#define __TIZEN_NNSTREAMER_API_MAIN_H__

#include <tizen_error.h>
#include <nnstreamer/tensor_typedef.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNSTREAMER_PIPELINE_MODULE
 * @{
 */

/**
 * @brief A handle of an NNStreamer pipeline.
 * @since_tizen 5.5
 */
typedef struct nns_pipeline *nns_pipeline_h;

/**
 * @brief A handle of a "sink node" of an NNStreamer pipeline
 * @since_tizen 5.5
 */
typedef struct nns_sink *nns_sink_h;

/**
 * @brief A handle of a "src node" of an NNStreamer pipeline
 * @since_tizen 5.5
 */
typedef struct nns_src *nns_src_h;

/**
 * @brief A handle of a "switch" of an NNStreamer pipeline
 * @since_tizen 5.5
 */
typedef struct nns_switch *nns_switch_h;

/**
 * @brief A handle of a "valve node" of an NNStreamer pipeline
 * @since_tizen 5.5
 */
typedef struct nns_valve *nns_valve_h;

/**
 * @brief Enumeration for the error codes of NNStreamer Pipelines.
 * @since_tizen 5.5
 * @todo The list is to be filled! (NYI)
 */
typedef enum {
  NNS_ERROR_NONE				= TIZEN_ERROR_NONE, /**< Success! */
  NNS_ERROR_INVALID_PARAMETER			= TIZEN_ERROR_INVALID_PARAMETER, /**< Invalid parameter */
  NNS_ERROR_NOT_SUPPORTED			= TIZEN_ERROR_NOT_SUPPORTED, /**< The feature is not supported */
  NNS_ERROR_PIPELINE_FAIL			= TIZEN_ERROR_STREAMS_PIPE, /**< Cannot create Gstreamer pipeline. */
} nns_error_e;

/**
 * @brief Enumeration for buffer deallocation policies.
 * @since_tizen 5.5
 * @todo The list is to be filled! (NYI)
 */
typedef enum {
  NNS_BUF_FREE_BY_NNSTREAMER			= 0, /**< Default. Application should not deallocate this buffer. NNStreamer will deallocate when the buffer is no more needed */
  NNS_BUF_DO_NOT_FREE				= 1, /**< This buffer is not to be freed. */
} nns_buf_policy;

/**
 * @brief Enumeration for nns pipeline state
 * @detail Refer to https://gstreamer.freedesktop.org/documentation/design/states.html
 *         The state diagram of pipeline looks like this, assuming that there is no errors.
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
 * @since_tizen 5.5
 * @todo The list is to be filled! (NYI) Fill with GST's state values.
 * @todo Check the consistency against MMFW APIs.
 */
typedef enum {
  NNS_PIPELINE_UNKNOWN				= 0, /**< Unknown state. Maybe not constructed? */
  NNS_PIPELINE_NULL				= 1, /**< GST-State "Null" */
  NNS_PIPELINE_READY				= 2, /**< GST-State "Ready" */
  NNS_PIPELINE_PAUSED				= 3, /**< GST-State "Paused" */
  NNS_PIPELINE_PLAYING				= 4, /**< GST-State "Playing" */
} nns_pipeline_state;

/**
 * @brief Enumeration for switch types
 * @detail This designates different GStreamer filters, "GstInputSelector"/"GetOutputSelector".
 * @since_tizen 5.5
 * @todo There may be more filters that can be supported.
 */
typedef enum {
  NNS_SWITCH_OUTPUTSELECTOR			= 0, /** GstOutputSelector */
  NNS_SWITCH_INPUTSELECTOR			= 1, /** GstInputSelector */
} nns_switch_type;

/**
 * @brief Callback for sink (tensor_sink) of nnstreamer pipelines (nnstreamer's output)
 * @detail If an application wants to accept data outputs of an nnstreamer stream, use this callback to get data from the stream. Note that the buffer may be deallocated after the return and this is synchronously called. Thus, if you need the data afterwards, copy the data to another buffer and return fast. Do not hold too much time in the callback. It is recommended to use very small tensors at sinks.
 * @since_tizen 5.5
 * @param[in] buf The contents of the tensor output (a single frame. tensor/tensors). Number of buf is determined by tensorsinfo->num_tensors.
 * @param[in] size The size of the buffer. Number of size is determined by tensorsinfo->num_tensors.
 * @param[in] tensorsinfo The cardinality, dimension, and type of given tensor/tensors.
 * @param[in/out] pdata User Application's Private Data
 */
typedef void (*nns_sink_cb)
(const char *buf[], const size_t size[], const GstTensorsInfo *tensorsinfo, void *pdata);

/****************************************************
 ** NNStreamer Pipeline Construction (gst-parse)   **
 ****************************************************/
/**
 * @brief Construct the pipeline (GStreamer + NNStreamer)
 * @detail Use this function to create a gst_parse_launch compatible NNStreamer pipelines.
 * @since_tizen 5.5
 * @param[in] pipeline_description The pipeline description compatible with GStreamer gst_parse_launch(). Refer to GStreamer manual or NNStreamer (github.com/nnsuite/nnstreamer) documentation for examples and the grammar.
 * @param[out] pipe The nnstreamer pipeline handler from the given description
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_construct (const char *pipeline_description, nns_pipeline_h *pipe);

/**
 * @brief Destroy the pipeline
 * @detail Use this function to destroy the pipeline constructed with nns_construct_pipeline.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be destroyed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_destroy (nns_pipeline_h pipe);

/**
 * @brief Get the state of pipeline
 * @detail Get the state of The pipeline handle returned by nns_construct_pipeline (pipe).
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be monitored.
 * @param[out] state The pipeline state.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_getstate (nns_pipeline_h pipe, nns_pipeline_state *state);

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Start the pipeline
 * @detail The pipeline handle returned by nns_construct_pipeline (pipe) is started.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be started.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_start (nns_pipeline_h pipe);

/**
 * @brief Stop the pipeline
 * @detail The pipeline handle returned by nns_construct_pipeline (pipe) is stopped.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be stopped.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_stop (nns_pipeline_h pipe);

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Register a callback for sink (tensor_sink) of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be attached with a sink node.
 * @param[in] sinkname The name of sink node, described with nns_pipeline_construct().
 * @param[in] cb The function to be called by the sink node.
 * @param[out] h The sink handle.
 * @param[in] pdata Private data for the callback. This value is passed to the callback when it's invoked.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_sink_register
(nns_pipeline_h pipe, const char *sinkname, nns_sink_cb cb, nns_sink_h *h, void *pdata);

/**
 * @brief Unregister a callback for sink (tensor_sink) of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] The nns_sink_handle to be unregistered (destroyed)
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_sink_unregister (nns_sink_h h);

/**
 * @brief Get a handle to operate as a src node of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be attached with a src node.
 * @param[in] srcname The name of src node, described with nns_pipeline_construct().
 * @param[out] tensorsinfo The cardinality, dimension, and type of given tensor/tensors.
 * @param[out] h The src handle.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_src_gethandle
(nns_pipeline_h pipe, const char *srcname, GstTensorsInfo *tensorsinfo, nns_src_h *h);

/**
 * @brief Close the given handle of a src node of nnstreamer pipelines.
 * @since_tizen 5.5
 * @param[in] h The src handle to be put (closed).
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_src_puthandle (nns_src_h h);

/**
 * @brief Put an input data frame.
 * @param[in] h The nns_src_handle returned by nns_pipeline_gethandle().
 * @param[out] policy The policy of buf deallocation.
 * @param[in] buf The input buffer, in the format of tensorsinfo given by nns_pipeline_gethandle()
 * @param[in] size The size of input buffer. This must be consistent with the given tensorsinfo.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_src_inputdata (nns_src_h h,
    nns_buf_policy policy, char *buf, size_t size);

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Get a handle to operate a "GstInputSelector / GstOutputSelector" node of nnstreamer pipelines
 * @detail Refer to gstreamer.freedesktop.org/data/doc/gstreamer/head/gst-plugins-bad-plugins/html/gst-plugins-bad-plugins-input-selector.html for input selectors
 *         Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-output-selector.html for output selectors
 * @param[in] pipe The pipeline to be managed.
 * @param[in] switchname The name of switch (InputSelector/OutputSelector)
 * @param[out] num_nodes The number of nodes selected by this switch node. We assume they are listed from 0 ... num_nodes-1.
 * @param[in] type The type of the switch.
 * @param[out] h The switch handle.
 * @return 0 on success (buf is filled). otherwise a negative error value.
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_switch_gethandle
(nns_pipeline_h pipe, const char *switchname, int *num_nodes, nns_switch_type type, nns_switch_h *h);

/**
 * @brief Close the given switch handle
 * @param[in] h The handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_switch_puthandle (nns_switch_h h);

/**
 * @brief Control the switch with the given handle to select input/output nodes(pads).
 * @param[in] h The switch handle returned by nns_pipeline_switch_gethandle()
 * @param[in] node The node number, which is 0 ... num_nodes-1.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_switch_select (nns_switch_h h, int node);

/**
 * @brief Get a handle to operate a "GstValve" node of nnstreamer pipelines
 * @detail Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-valve.html for more info.
 * @param[in] pipe The pipeline to be managed.
 * @param[in] valvename The name of valve (Valve)
 * @param[out] h The valve handle.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_valve_gethandle
(nns_pipeline_h pipe, const char *valvename, nns_valve_h *h);

/**
 * @brief Close the given valve handle
 * @param[in] h The handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_valve_puthandle (nns_valve_h h);

/**
 * @brief Control the valve with the given handle
 * @param[in] h The valve handle returned by nns_pipeline_valve_gethandle()
 * @param[in] valve_open 0 to close (stop flow). 1 to open (continue flow)
 * @return @c 0 on success. otherwise a negative error value
 * @retval #NNS_ERROR_NONE Successful
 */
int nns_pipeline_valve_control (nns_valve_h h, int valve_open);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /*__TIZEN_NNSTREAMER_API_MAIN_H__*/
