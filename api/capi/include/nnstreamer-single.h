/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file nnstreamer-single.h
 * @date 29 March 2019
 * @brief NNStreamer single-shot invocation C-API Header.
 *        This allows to invoke a neural network model directly.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * @details This is targeting Tizen 5.5 M2.
 */

#ifndef __TIZEN_MACHINELEARNING_NNSTREAMER_SINGLE_H__
#define __TIZEN_MACHINELEARNING_NNSTREAMER_SINGLE_H__

#include <nnstreamer.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNSTREAMER_SINGLE_MODULE
 * @{
 */

/**
 * @brief A handle of a single-shot instance.
 * @since_tizen 5.5
 */
typedef void *ml_single_h;

/*************
 * MAIN FUNC *
 *************/
/**
 * @brief Opens an ML model and returns the instance as a handle.
 * @details Even if the model has flexible input data dimensions,
 *          input data frames of an instance of a model should share the same dimension.
 * @since_tizen 5.5
 * @remarks http://tizen.org/privilege/mediastorage is needed if @a model is relevant to media storage.
 * @remarks http://tizen.org/privilege/externalstorage is needed if @a model is relevant to external storage.
 * @param[out] single This is the model handle opened. Users are required to close
 *                   the given instance with ml_single_close().
 * @param[in] model This is the path to the neural network model file.
 * @param[in] input_info This is required if the given model has flexible input
 *                      dimension, where the input dimension MUST be given
 *                      before executing the model.
 *                      However, once it's given, the input dimension cannot
 *                      be changed for the given model handle.
 *                      It is required by some custom filters of NNStreamer.
 *                      You may set NULL if it's not required.
 * @param[in] output_info This is required if the given model has flexible output dimension.
 * @param[in] nnfw The neural network framework used to open the given @a model.
 *                 Set #ML_NNFW_TYPE_ANY to let it auto-detect.
 * @param[in] hw Tell the corresponding @a nnfw to use a specific hardware.
 *               Set #ML_NNFW_HW_ANY if it does not matter.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Failed to start the pipeline.
 * @retval #ML_ERROR_PERMISSION_DENIED The application does not have the privilege to access to the media storage or external storage.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_single_open (ml_single_h *single, const char *model, const ml_tensors_info_h input_info, const ml_tensors_info_h output_info, ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw);

/**
 * @brief Closes the opened model handle.
 * @since_tizen 5.5
 * @param[in] single The model handle to be closed.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid (Pipeline is not negotiated yet.)
 */
int ml_single_close (ml_single_h single);

/**
 * @brief Invokes the model with the given input data.
 * @details Even if the model has flexible input data dimensions,
 *          input data frames of an instance of a model should share the same dimension.
 *          Note that this has a default timeout of 3 seconds. If an application wants to change the time to wait for an output, set the timeout using ml_single_set_timeout().
 * @since_tizen 5.5
 * @param[in] single The model handle to be inferred.
 * @param[in] input The input data to be inferred.
 * @param[out] output The allocated output buffer. The caller is responsible for freeing the output buffer with ml_tensors_data_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Failed to push a buffer into source element.
 * @retval #ML_ERROR_TIMED_OUT Failed to get the result from sink element.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_single_invoke (ml_single_h single, const ml_tensors_data_h input, ml_tensors_data_h *output);

/**
 * @brief Invokes the model with the given input data with the given info.
 * @details This function changes the input data dimensions for the model.
 *          Note that this has a default timeout of 3 seconds. If an application wants to change the time to wait for an output, set the timeout using ml_single_set_timeout().
 * @since_tizen 6.0
 * @param[in] single The model handle to be inferred.
 * @param[in] input The input data to be inferred.
 * @param[in] in_info The handle of input tensors information.
 * @param[out] output The allocated output buffer. The caller is responsible for freeing the output buffer with ml_tensors_data_destroy().
 * @param[out] out_info The handle of output tensors information. The caller is responsible for freeing the information with ml_tensors_info_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 * @retval #ML_ERROR_STREAMS_PIPE Failed to push a buffer into source element.
 * @retval #ML_ERROR_TIMED_OUT Failed to get the result from sink element.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_single_invoke_dynamic (ml_single_h single, const ml_tensors_data_h input, const ml_tensors_info_h in_info, ml_tensors_data_h *output, ml_tensors_info_h *out_info);

/*************
 * UTILITIES *
 *************/

/**
 * @brief Gets the information (tensor dimension, type, name and so on) of required input data for the given model.
 * @details Note that a model may not have such information if its input type is flexible.
 *          The names of tensors are sometimes unavailable (optional), while its dimensions and types are always available.
 * @since_tizen 5.5
 * @param[in] single The model handle.
 * @param[out] info The handle of input tensors information. The caller is responsible for freeing the information with ml_tensors_info_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 */
int ml_single_get_input_info (ml_single_h single, ml_tensors_info_h *info);

/**
 * @brief Gets the information (tensor dimension, type, name and so on) of output data for the given model.
 * @details Note that a model may not have such information if its output type is flexible and output type is not determined statically.
 *          The names of tensors are sometimes unavailable (optional), while its dimensions and types are always available.
 * @since_tizen 5.5
 * @param[in] single The model handle.
 * @param[out] info The handle of output tensors information. The caller is responsible for freeing the information with ml_tensors_info_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 */
int ml_single_get_output_info (ml_single_h single, ml_tensors_info_h *info);

/**
 * @brief Sets the information (tensor dimension, type, name and so on) of required input data for the given model.
 * @details Note that a model/framework may not support setting such information.
 * @since_tizen 6.0
 * @param[in] single The model handle.
 * @param[in] info The handle of input tensors information.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED This implies that the given framework does not support dynamic dimensions.
 *         Use ml_single_get_input_info() and ml_single_get_output_info() instead for this framework.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 */
int ml_single_set_input_info (ml_single_h single, const ml_tensors_info_h info);

/**
 * @brief Sets the maximum amount of time to wait for an output, in milliseconds.
 * @since_tizen 5.5
 * @param[in] single The model handle.
 * @param[in] timeout The time to wait for an output.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 */
int ml_single_set_timeout (ml_single_h single, unsigned int timeout);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNSTREAMER_SINGLE_H__ */
