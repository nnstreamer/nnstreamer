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
 * @brief NNStreamer/Pipeline(main) C-API Header.
 *        This allows to construct and control NNStreamer pipelines.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNSTREAMER_H__
#define __TIZEN_MACHINELEARNING_NNSTREAMER_H__

#include <stddef.h>
#include <stdbool.h>

#include <tizen_error.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNSTREAMER_PIPELINE_MODULE
 * @{
 */

/**
 * @brief The virtual name to set the video source of camcorder in Tizen.
 * @details If an application needs to access the camera device to construct the pipeline, set the virtual name as a video source element.
 *          Note that you have to add 'http://tizen.org/privilege/camera' into the manifest of your application.
 * @since_tizen 5.5
 */
#define ML_TIZEN_CAM_VIDEO_SRC "tizencamvideosrc"

/**
 * @brief The virtual name to set the audio source of camcorder in Tizen.
 * @details If an application needs to access the recorder device to construct the pipeline, set the virtual name as an audio source element.
 *          Note that you have to add 'http://tizen.org/privilege/recorder' into the manifest of your application.
 * @since_tizen 5.5
 */
#define ML_TIZEN_CAM_AUDIO_SRC "tizencamaudiosrc"

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
 * @brief The dimensions of a tensor that NNStreamer supports.
 * @since_tizen 5.5
 */
typedef unsigned int ml_tensor_dimension[ML_TENSOR_RANK_LIMIT];

/**
 * @brief A handle of a tensors metadata instance.
 * @since_tizen 5.5
 */
typedef void *ml_tensors_info_h;

/**
 * @brief A handle of input or output frames. #ml_tensors_info_h is the handle for tensors metadata.
 * @since_tizen 5.5
 */
typedef void *ml_tensors_data_h;

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
 * @brief Types of NNFWs.
 * @since_tizen 5.5
 */
typedef enum {
  ML_NNFW_TYPE_ANY = 0, /**< NNHW is not specified (Try to determine the NNFW with file extension). */
  ML_NNFW_TYPE_CUSTOM_FILTER, /**< Custom filter (Independent shared object). */
  ML_NNFW_TYPE_TENSORFLOW_LITE, /**< Tensorflow-lite (.tflite). */
  ML_NNFW_TYPE_TENSORFLOW, /**< Tensorflow (.pb). */
  ML_NNFW_TYPE_NNFW, /**< Neural Network Inference framework, which is developed by SR */
} ml_nnfw_type_e;

/**
 * @brief Types of hardware resources to be used for NNFWs. Note that if the affinity (nnn) is not supported by the driver or hardware, it is ignored.
 * @since_tizen 5.5
 */
typedef enum {
  ML_NNFW_HW_ANY = 0,      /**< Hardware resource is not specified. */
  ML_NNFW_HW_AUTO = 1,     /**< Try to schedule and optimize if possible. */
  ML_NNFW_HW_CPU = 0x1000, /**< 0x1000: any CPU. 0x1nnn: CPU # nnn-1. */
  ML_NNFW_HW_GPU = 0x2000, /**< 0x2000: any GPU. 0x2nnn: GPU # nnn-1. */
  ML_NNFW_HW_NPU = 0x3000, /**< 0x3000: any NPU. 0x3nnn: NPU # nnn-1. */
} ml_nnfw_hw_e;

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
  ML_TENSOR_TYPE_UNKNOWN         /**< Unknown type */
} ml_tensor_type_e;

/**
 * @brief Enumeration for the error codes of NNStreamer Pipelines.
 * @since_tizen 5.5
 */
typedef enum {
  ML_ERROR_NONE                 = TIZEN_ERROR_NONE, /**< Success! */
  ML_ERROR_INVALID_PARAMETER    = TIZEN_ERROR_INVALID_PARAMETER, /**< Invalid parameter */
  ML_ERROR_STREAMS_PIPE         = TIZEN_ERROR_STREAMS_PIPE, /**< Cannot create or access GStreamer pipeline. */
  ML_ERROR_TRY_AGAIN            = TIZEN_ERROR_TRY_AGAIN, /**< The pipeline is not ready, yet (not negotiated, yet) */
  ML_ERROR_UNKNOWN              = TIZEN_ERROR_UNKNOWN,  /**< Unknown error */
  ML_ERROR_TIMED_OUT            = TIZEN_ERROR_TIMED_OUT,  /**< Time out */
  ML_ERROR_NOT_SUPPORTED        = TIZEN_ERROR_NOT_SUPPORTED, /**< The feature is not supported */
  ML_ERROR_PERMISSION_DENIED    = TIZEN_ERROR_PERMISSION_DENIED, /**< Permission denied */
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
 * @details The pipeline state is described on @ref CAPI_ML_NNSTREAMER_PIPELINE_STATE_DIAGRAM.
 * Refer to https://gstreamer.freedesktop.org/documentation/plugin-development/basics/states.html.
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_STATE_UNKNOWN				= 0, /**< Unknown state. Maybe not constructed? */
  ML_PIPELINE_STATE_NULL				= 1, /**< GST-State "Null" */
  ML_PIPELINE_STATE_READY				= 2, /**< GST-State "Ready" */
  ML_PIPELINE_STATE_PAUSED				= 3, /**< GST-State "Paused" */
  ML_PIPELINE_STATE_PLAYING				= 4, /**< GST-State "Playing" */
} ml_pipeline_state_e;

/**
 * @brief Enumeration for switch types.
 * @details This designates different GStreamer filters, "GstInputSelector"/"GetOutputSelector".
 * @since_tizen 5.5
 */
typedef enum {
  ML_PIPELINE_SWITCH_OUTPUT_SELECTOR			= 0, /**< GstOutputSelector */
  ML_PIPELINE_SWITCH_INPUT_SELECTOR			= 1, /**< GstInputSelector */
} ml_pipeline_switch_e;

/**
 * @brief Callback for sink element of NNStreamer pipelines (pipeline's output).
 * @details If an application wants to accept data outputs of an NNStreamer stream, use this callback to get data from the stream. Note that the buffer may be deallocated after the return and this is synchronously called. Thus, if you need the data afterwards, copy the data to another buffer and return fast. Do not spend too much time in the callback. It is recommended to use very small tensors at sinks.
 * @since_tizen 5.5
 * @remarks The @a data can be used only in the callback. To use outside, make a copy.
 * @remarks The @a info can be used only in the callback. To use outside, make a copy.
 * @param[out] data The handle of the tensor output (a single frame. tensor/tensors). Number of tensors is determined by ml_tensors_info_get_count() with the handle 'info'. Note that max num_tensors is 16 (#ML_TENSOR_SIZE_LIMIT).
 * @param[out] info The handle of tensors information (cardinality, dimension, and type of given tensor/tensors).
 * @param[out] user_data User application's private data.
 */
typedef void (*ml_pipeline_sink_cb) (const ml_tensors_data_h data, const ml_tensors_info_h info, void *user_data);

/**
 * @brief Callback for the change of pipeline state.
 * @details If an application wants to get the change of pipeline state, use this callback. This callback can be registered when constructing the pipeline using ml_pipeline_construct(). Do not spend too much time in the callback.
 * @since_tizen 5.5
 * @param[out] state The new state of the pipeline.
 * @param[out] user_data User application's private data.
 */
typedef void (*ml_pipeline_state_cb) (ml_pipeline_state_e state, void *user_data);

/****************************************************
 ** NNStreamer Pipeline Construction (gst-parse)   **
 ****************************************************/
/**
 * @brief Constructs the pipeline (GStreamer + NNStreamer).
 * @details Use this function to create gst_parse_launch compatible NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a pipe handle must be released using ml_pipeline_destroy().
 * @remarks http://tizen.org/privilege/mediastorage is needed if @a pipeline_description is relevant to media storage.
 * @remarks http://tizen.org/privilege/externalstorage is needed if @a pipeline_description is relevant to external storage.
 * @remarks http://tizen.org/privilege/camera is needed if @a pipeline_description accesses the camera device.
 * @remarks http://tizen.org/privilege/recorder is needed if @a pipeline_description accesses the recorder device.
 * @param[in] pipeline_description The pipeline description compatible with GStreamer gst_parse_launch(). Refer to GStreamer manual or NNStreamer (https://github.com/nnsuite/nnstreamer) documentation for examples and the grammar.
 * @param[in] cb The function to be called when the pipeline state is changed. You may set NULL if it's not required.
 * @param[in] user_data Private data for the callback. This value is passed to the callback when it's invoked.
 * @param[out] pipe The NNStreamer pipeline handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Pipeline construction is failed because of wrong parameter or initialization failure.
 * @retval #ML_ERROR_PERMISSION_DENIED The application does not have the required privilege to access to the media storage, external storage, microphone, or camera.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_UNKNOWN or #ML_PIPELINE_STATE_NULL.
 * @post The pipeline state will be #ML_PIPELINE_STATE_PAUSED in the same thread.
 */
int ml_pipeline_construct (const char *pipeline_description, ml_pipeline_state_cb cb, void *user_data, ml_pipeline_h *pipe);

/**
 * @brief Destroys the pipeline.
 * @details Use this function to destroy the pipeline constructed with ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be destroyed.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER The parameter is invalid (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Cannot access the pipeline status.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PLAYING or #ML_PIPELINE_STATE_PAUSED.
 * @post The pipeline state will be #ML_PIPELINE_STATE_NULL.
 */
int ml_pipeline_destroy (ml_pipeline_h pipe);

/**
 * @brief Gets the state of pipeline.
 * @details Gets the state of the pipeline handle returned by ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline handle.
 * @param[out] state The pipeline state.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to get state from the pipeline.
 */
int ml_pipeline_get_state (ml_pipeline_h pipe, ml_pipeline_state_e *state);

/****************************************************
 ** NNStreamer Pipeline Start/Stop Control         **
 ****************************************************/
/**
 * @brief Starts the pipeline, asynchronously.
 * @details The pipeline handle returned by ml_pipeline_construct() is started.
 *          Note that this is asynchronous function. State might be "pending".
 *          If you need to get the changed state, add a callback while constructing a pipeline with ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (NPipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to start the pipeline.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PAUSED.
 * @post The pipeline state will be #ML_PIPELINE_STATE_PLAYING.
 */
int ml_pipeline_start (ml_pipeline_h pipe);

/**
 * @brief Stops the pipeline, asynchronously.
 * @details The pipeline handle returned by ml_pipeline_construct() is stopped.
 *          Note that this is asynchronous function. State might be "pending".
 *          If you need to get the changed state, add a callback while constructing a pipeline with ml_pipeline_construct().
 * @since_tizen 5.5
 * @param[in] pipe The pipeline to be stopped.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to stop the pipeline.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PLAYING.
 * @post The pipeline state will be #ML_PIPELINE_STATE_PAUSED.
 */
int ml_pipeline_stop (ml_pipeline_h pipe);

/****************************************************
 ** NNStreamer Pipeline Sink/Src Control           **
 ****************************************************/
/**
 * @brief Registers a callback for sink node of NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a sink_handle handle must be unregistered using ml_pipeline_sink_unregister().
 * @param[in] pipe The pipeline to be attached with a sink node.
 * @param[in] sink_name The name of sink node, described with ml_pipeline_construct().
 * @param[in] cb The function to be called by the sink node.
 * @param[in] user_data Private data for the callback. This value is passed to the callback when it's invoked.
 * @param[out] sink_handle The sink handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid. (Not negotiated, sink_name is not found, or sink_name has an invalid type.)
 * @retval #ML_ERROR_STREAMS_PIPE Failed to connect a signal to sink element.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PAUSED.
 */
int ml_pipeline_sink_register (ml_pipeline_h pipe, const char *sink_name, ml_pipeline_sink_cb cb, void *user_data, ml_pipeline_sink_h *sink_handle);

/**
 * @brief Unregisters a callback for sink node of NNStreamer pipelines.
 * @since_tizen 5.5
 * @param[in] sink_handle The sink handle to be unregistered.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 *
 * @pre The pipeline state should be #ML_PIPELINE_STATE_PAUSED.
 */
int ml_pipeline_sink_unregister (ml_pipeline_sink_h sink_handle);

/**
 * @brief Gets a handle to operate as a src node of NNStreamer pipelines.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a src_handle handle must be released using ml_pipeline_src_release_handle().
 * @param[in] pipe The pipeline to be attached with a src node.
 * @param[in] src_name The name of src node, described with ml_pipeline_construct().
 * @param[out] src_handle The src handle.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Fail to get SRC element.
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int ml_pipeline_src_get_handle (ml_pipeline_h pipe, const char *src_name, ml_pipeline_src_h *src_handle);

/**
 * @brief Releases the given src handle.
 * @since_tizen 5.5
 * @param[in] src_handle The src handle to be released.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_src_release_handle (ml_pipeline_src_h src_handle);

/**
 * @brief Adds an input data frame.
 * @since_tizen 5.5
 * @param[in] src_handle The source handle returned by ml_pipeline_src_get_handle().
 * @param[in] data The handle of input tensors, in the format of tensors info given by ml_pipeline_src_get_tensors_info().
 *                 This function takes ownership of the data if @a policy is #ML_PIPELINE_BUF_POLICY_AUTO_FREE.
 * @param[in] policy The policy of buf deallocation.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The pipeline has inconsistent padcaps. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int ml_pipeline_src_input_data (ml_pipeline_src_h src_handle, ml_tensors_data_h data, ml_pipeline_buf_policy_e policy);

/**
 * @brief Gets a handle for the tensors information of given src node.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a info handle must be released using ml_tensors_info_destroy().
 * @param[in] src_handle The source handle returned by ml_pipeline_src_get_handle().
 * @param[out] info The handle of tensors information.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The pipeline has inconsistent padcaps. (Pipeline is not negotiated yet.)
 * @retval #ML_ERROR_TRY_AGAIN The pipeline is not ready yet.
 */
int ml_pipeline_src_get_tensors_info (ml_pipeline_src_h src_handle, ml_tensors_info_h *info);

/****************************************************
 ** NNStreamer Pipeline Switch/Valve Control       **
 ****************************************************/

/**
 * @brief Gets a handle to operate a "GstInputSelector / GstOutputSelector" node of NNStreamer pipelines.
 * @details Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-input-selector.html for input selectors.
 *          Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-output-selector.html for output selectors.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a switch_handle handle must be released using ml_pipeline_switch_release_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] switch_name The name of switch (InputSelector/OutputSelector).
 * @param[out] switch_type The type of the switch. If NULL, it is ignored.
 * @param[out] switch_handle The switch handle.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_get_handle (ml_pipeline_h pipe, const char *switch_name, ml_pipeline_switch_e *switch_type, ml_pipeline_switch_h *switch_handle);

/**
 * @brief Releases the given switch handle.
 * @since_tizen 5.5
 * @param[in] switch_handle The handle to be released.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_release_handle (ml_pipeline_switch_h switch_handle);

/**
 * @brief Controls the switch with the given handle to select input/output nodes(pads).
 * @since_tizen 5.5
 * @param[in] switch_handle The switch handle returned by ml_pipeline_switch_get_handle().
 * @param[in] pad_name The name of the chosen pad to be activated. Use ml_pipeline_switch_get_pad_list() to list the available pad names.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_switch_select (ml_pipeline_switch_h switch_handle, const char *pad_name);

/**
 * @brief Gets the pad names of a switch.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a list and its contents should be released using g_free(). Refer the below sample code.
 * @param[in] switch_handle The switch handle returned by ml_pipeline_switch_get_handle().
 * @param[out] list NULL terminated array of char*. The caller must free each string (char*) in the list and free the list itself.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE The element is not both input and output switch (Internal data inconsistency).
 *
 * Here is an example of the usage:
 * @code
 * int status;
 * gchar *pipeline;
 * ml_pipeline_h handle;
 * ml_pipeline_switch_e switch_type;
 * ml_pipeline_switch_h switch_handle;
 * gchar **node_list = NULL;
 *
 * // pipeline description
 * pipeline = g_strdup ("videotestsrc is-live=true ! videoconvert ! tensor_converter ! output-selector name=outs "
 *     "outs.src_0 ! tensor_sink name=sink0 async=false "
 *     "outs.src_1 ! tensor_sink name=sink1 async=false");
 *
 * status = ml_pipeline_construct (pipeline, NULL, NULL, &handle);
 * if (status != ML_ERROR_NONE) {
 *   // handle error case
 *   goto error;
 * }
 *
 * status = ml_pipeline_switch_get_handle (handle, "outs", &switch_type, &switch_handle);
 * if (status != ML_ERROR_NONE) {
 *   // handle error case
 *   goto error;
 * }
 *
 * status = ml_pipeline_switch_get_pad_list (switch_handle, &node_list);
 * if (status != ML_ERROR_NONE) {
 *   // handle error case
 *   goto error;
 * }
 *
 * if (node_list) {
 *   gchar *name = NULL;
 *   guint idx = 0;
 *
 *   while ((name = node_list[idx++]) != NULL) {
 *     // node name is 'src_0' or 'src_1'
 *
 *     // release name
 *     g_free (name);
 *   }
 *   // release list of switch pads
 *   g_free (node_list);
 * }
 *
 * error:
 * ml_pipeline_switch_release_handle (switch_handle);
 * ml_pipeline_destroy (handle);
 * g_free (pipeline);
 * @endcode
 */
int ml_pipeline_switch_get_pad_list (ml_pipeline_switch_h switch_handle, char ***list);

/**
 * @brief Gets a handle to operate a "GstValve" node of NNStreamer pipelines.
 * @details Refer to https://gstreamer.freedesktop.org/data/doc/gstreamer/head/gstreamer-plugins/html/gstreamer-plugins-valve.html for more info.
 * @since_tizen 5.5
 * @remarks If the function succeeds, @a valve_handle handle must be released using ml_pipeline_valve_release_handle().
 * @param[in] pipe The pipeline to be managed.
 * @param[in] valve_name The name of valve (Valve).
 * @param[out] valve_handle The valve handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_valve_get_handle (ml_pipeline_h pipe, const char *valve_name, ml_pipeline_valve_h *valve_handle);

/**
 * @brief Releases the given valve handle.
 * @since_tizen 5.5
 * @param[in] valve_handle The handle to be released.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_valve_release_handle (ml_pipeline_valve_h valve_handle);

/**
 * @brief Controls the valve with the given handle.
 * @since_tizen 5.5
 * @param[in] valve_handle The valve handle returned by ml_pipeline_valve_get_handle().
 * @param[in] open @c true to open(let the flow pass), @c false to close (drop & stop the flow).
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_pipeline_valve_set_open (ml_pipeline_valve_h valve_handle, bool open);

/****************************************************
 ** NNStreamer Utilities                           **
 ****************************************************/
/**
 * @brief Creates a tensors information handle with default value.
 * @since_tizen 5.5
 * @param[out] info The handle of tensors information.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_create (ml_tensors_info_h *info);

/**
 * @brief Frees the given handle of a tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @return 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_destroy (ml_tensors_info_h info);

/**
 * @brief Validates the given tensors information.
 * @details If the function returns an error, @a valid is not changed.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information to be validated.
 * @param[out] valid @c true if it's valid, @c false if if it's invalid.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_validate (const ml_tensors_info_h info, bool *valid);

/**
 * @brief Copies the tensors information.
 * @since_tizen 5.5
 * @param[out] dest A destination handle of tensors information.
 * @param[in] src The tensors information to be copied.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_clone (ml_tensors_info_h dest, const ml_tensors_info_h src);

/**
 * @brief Sets the number of tensors with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] count The number of tensors.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_set_count (ml_tensors_info_h info, unsigned int count);

/**
 * @brief Gets the number of tensors with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[out] count The number of tensors.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_get_count (ml_tensors_info_h info, unsigned int *count);

/**
 * @brief Sets the tensor name with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor to be updated.
 * @param[in] name The tensor name to be set.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_set_tensor_name (ml_tensors_info_h info, unsigned int index, const char *name);

/**
 * @brief Gets the tensor name with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor.
 * @param[out] name The tensor name.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_get_tensor_name (ml_tensors_info_h info, unsigned int index, char **name);

/**
 * @brief Sets the tensor type with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor to be updated.
 * @param[in] type The tensor type to be set.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_set_tensor_type (ml_tensors_info_h info, unsigned int index, const ml_tensor_type_e type);

/**
 * @brief Gets the tensor type with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor.
 * @param[out] type The tensor type.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_get_tensor_type (ml_tensors_info_h info, unsigned int index, ml_tensor_type_e *type);

/**
 * @brief Sets the tensor dimension with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor to be updated.
 * @param[in] dimension The tensor dimension to be set.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_set_tensor_dimension (ml_tensors_info_h info, unsigned int index, const ml_tensor_dimension dimension);

/**
 * @brief Gets the tensor dimension with given handle of tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor.
 * @param[out] dimension The tensor dimension.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_get_tensor_dimension (ml_tensors_info_h info, unsigned int index, ml_tensor_dimension dimension);

/**
 * @brief Gets the size of tensors data in the given tensors information handle in bytes.
 * @details If an application needs to get the total byte size of tensors, set the @a index '-1'. Note that the maximum number of tensors is 16 (#ML_TENSOR_SIZE_LIMIT).
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information.
 * @param[in] index The index of the tensor.
 * @param[out] data_size The byte size of tensor data.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_info_get_tensor_size (ml_tensors_info_h info, int index, size_t *data_size);

/**
 * @brief Creates a tensor data frame with the given tensors information.
 * @since_tizen 5.5
 * @param[in] info The handle of tensors information for the allocation.
 * @param[out] data The handle of tensors data. The caller is responsible for freeing the allocated data with ml_tensors_data_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Failed to allocate new memory.
 */
int ml_tensors_data_create (const ml_tensors_info_h info, ml_tensors_data_h *data);

/**
 * @brief Frees the given tensors' data handle.
 * @since_tizen 5.5
 * @param[in] data The handle of tensors data.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_data_destroy (ml_tensors_data_h data);

/**
 * @brief Gets a tensor data of given handle.
 * @details This returns the pointer of memory block in the handle. Do not deallocate the returned tensor data.
 * @since_tizen 5.5
 * @param[in] data The handle of tensors data.
 * @param[in] index The index of the tensor.
 * @param[out] raw_data Raw tensor data in the handle.
 * @param[out] data_size Byte size of tensor data.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_data_get_tensor_data (ml_tensors_data_h data, unsigned int index, void **raw_data, size_t *data_size);

/**
 * @brief Copies a tensor data to given handle.
 * @since_tizen 5.5
 * @param[in] data The handle of tensors data.
 * @param[in] index The index of the tensor.
 * @param[in] raw_data Raw tensor data to be copied.
 * @param[in] data_size Byte size of raw data.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_tensors_data_set_tensor_data (ml_tensors_data_h data, unsigned int index, const void *raw_data, const size_t data_size);

/**
 * @brief Checks the availability of the given execution environments.
 * @details If the function returns an error, @a available is not changed.
 * @since_tizen 5.5
 * @param[in] nnfw Check if the nnfw is available in the system.
 *               Set #ML_NNFW_TYPE_ANY to skip checking nnfw.
 * @param[in] hw Check if the hardware is available in the system.
 *               Set #ML_NNFW_HW_ANY to skip checking hardware.
 * @param[out] available @c true if it's available, @c false if if it's not available.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful and the environments are available.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Given parameter is invalid.
 */
int ml_check_nnfw_availability (ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw, bool *available);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNSTREAMER_H__ */
