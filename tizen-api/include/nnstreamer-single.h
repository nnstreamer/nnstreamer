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
 * @file nnstreamer-single.h
 * @date 29 March 2019
 * @brief Tizen NNStreamer single shot invocation C-API Header.
 *        This allows to invoke a neural network model directly.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * @detail This is targetting Tizen 5.5 M2.
 */

#ifndef __TIZEN_NNSTREAMER_SINGLE_H__
#define __TIZEN_NNSTREAMER_SINGLE_H__

#include <stddef.h>
#include <tizen_error.h>
#include "nnstreamer.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNSTREAMER_SINGLE_MODULE
 * @{
 */

/**
 * @brief A handle of a simpleshot instance
 * @since_tizen 5.5
 */
typedef void *ml_simpleshot_model_h;

/**
 * @brief Types of NNFWs.
 * @since_tizen 5.5
 */
typedef enum {
  ML_NNFW_UNKNOWN = 0, /**< it is unknown or we do not care this value. */
  ML_NNFW_CUSTOM_FILTER, /**< custom filter (independent shared object). */
  ML_NNFW_TENSORFLOW_LITE, /**< tensorflow-lite (.tflite). */
  ML_NNFW_TENSORFLOW, /**< tensorflow (.pb). */
} ml_model_nnfw;

/**
 * @brief Types of NNFWs. Note that if the affinity (nnn) is not supported by the driver or hardware, it is ignored.
 * @since_tizen 5.5
 */
typedef enum {
  ML_NNFW_HW_DO_NOT_CARE = 0, /**< Hardware resource is not specified. */
  ML_NNFW_HW_AUTO = 1, /**< Try to schedule and optimize if possible. */
  ML_NNFW_HW_CPU = 0x1000, /**< 0x1000: any CPU. 0x1nnn: CPU # nnn-1. */
  ML_NNFW_HW_GPU = 0x2000, /**< 0x2000: any GPU. 0x2nnn: GPU # nnn-1. */
  ML_NNFW_HW_NPU = 0x3000, /**< 0x3000: any NPU. 0x3nnn: NPU # nnn-1. */
} ml_model_hw;

/*************
 * MAIN FUNC *
 *************/
/**
 * @brief Open an ML model and return the model as a handle.
 * @since_tizen 5.5
 * @param[in] model_path This is the path to the neural network model file.
 * @param[out] model This is the model opened. Users are required to close
 *                   the given model with ml_model_close().
 * @param[in] input_type This is required if the given model has flexible input
 *                      dimension, where the input dimension MUST be given
 *                      before executing the model.
 *                      However, once it's given, the input dimension cannot
 *                      be changed for the given model handle.
 *                      It is required by some custom filters of nnstreamer.
 *                      You may set NULL if it's not required.
 * @param[in] output_type This is required if the given model has flexible output dimension.
 * @param[in] nnfw The nerual network framework used to open the given
 *                 @model_path. Set ML_NNFW_UNKNOWN to let it auto-detect.
 * @param[in] hw Tell the corresponding @nnfw to use a specific hardware.
 *               Set ML_NNFW_HW_DO_NOT_CARE if it does not matter.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 *
 * @detail Even if the model has flexible input data dimensions,
 *         input data frames of an instance of a model should share the
 *         same dimension.
 */
int ml_model_open (const char *model_path, ml_simpleshot_model_h *model,
    const ml_tensors_info_s *input_type, const ml_tensors_info_s *output_type,
    ml_model_nnfw nnfw, ml_model_hw hw);

/**
 * @brief Close the opened model handle.
 * @since_tizen 5.5
 * @param[in] model The model handle to be closed.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid (pipe is NULL?)
 */
int ml_model_close (ml_simpleshot_model_h model);

/**
 * @brief Invoke the model with the given input data.
 * @since_tizen 5.5
 * @param[in] model The model to be inferred.
 * @param[in] input The input data to be inferred.
 * @param[out] output The output buffer. Set NULL if you want to let
 *                    this function to allocate a new output buffer.
 * @return @c The output buffer. If @output is NULL, this is a newly
 *         allocated buffer; thus, the user needs to free it.
 *         If there is an error, this is set NULL. Check get_last_result()
 *         of tizen_error.h in such cases.
 *
 * @detail Even if the model has flexible input data dimensions,
 *         input data frames of an instance of a model should share the
 *         same dimension.
 */
ml_tensors_data_s * ml_model_inference (ml_simpleshot_model_h model,
    const ml_tensors_data_s *input, ml_tensors_data_s *output);

/*************
 * UTILITIES *
 *************/

/**
 * @brief Get type (tensor dimension, type, name and so on) of required input
 *        data for the given model.
 * @detail Note that a model may not have such
 *         information if its input type is flexible.
 *         Besides, names of tensors may be not available while dimensions and
 *         types are available.
 * @since_tizen 5.5
 * @param[in] model The model to be investigated
 * @param[out] input_type The type of input tensor.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 */
int ml_model_get_input_type (ml_simpleshot_model_h model,
    ml_tensors_info_s *input_type);

/**
 * @brief Get type (tensor dimension, type, name and so on) of output
 *        data of the given model.
 * @detail Note that a model may not have such
 *         information if its input type is flexible and output type is
 *         not determined statically.
 *         Besides, names of tensors may be not available while dimensions and
 *         types are available.
 * @since_tizen 5.5
 * @param[in] model The model to be investigated
 * @param[out] output_type The type of output tensor.
 * @return @c 0 on success. otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful
 */
int ml_model_get_output_type (ml_simpleshot_model_h model,
    ml_tensors_info_s *output_type);

/**
 * @brief Get the byte size of the given tensor type.
 * @since_tizen 5.5
 * @param[in] info The tensor information to be investigated.
 * @return @c >= 0 on success with byte size. otherwise a negative error value
 */
size_t ml_util_get_tensor_size (const ml_tensor_info_s *info);

/**
 * @brief Get the byte size of the given tensors type.
 * @since_tizen 5.5
 * @param[in] info The tensors information to be investigated.
 * @return @c >= 0 on success with byte size. otherwise a negative error value
 */
size_t ml_util_get_tensors_size (const ml_tensors_info_s *info);

/**
 * @brief Free the tensors type pointer.
 * @since_tizen 5.5
 * @param[in] type the tensors type pointer to be freed.
 */
void ml_model_free_tensors_info (ml_tensors_info_s *type);

/**
 * @brief Free the tensors data pointer.
 * @since_tizen 5.5
 * @param[in] data The tensors data pointer to be freed.
 */
void ml_model_free_tensors_data (ml_tensors_data_s *data);

/**
 * @brief Allocate a tensor data frame with the given tensors type.
 * @since_tizen 5.5
 * @param[in] info The tensors information for the allocation
 * @return @c Tensors data pointer allocated. Null if error.
 * @retval NULL there is an error. call ml_get_last_result() to get specific error numbers.
 */
ml_tensors_data_s *ml_model_allocate_tensors_data (const ml_tensors_info_s *info);

/**
 * @brief Check the availability of the given execution environments.
 * @since_tizen 5.5
 * @param[in] nnfw Check if the nnfw is available in the system.
 *                 Set ML_NNFW_UNKNOWN to skip checking nnfw.
 * @param[in] hw Check if the hardware is available in the system.
 *               Set ML_NNFW_HW_DO_NOT_CARE to skip checking hardware.
 * @return @c 0 if it's available. 1 if it's not available.
 *            negative value if there is an error.
 * @retval #ML_ERROR_NONE Successful and the environments are available.
 * @retval 1 Successful but the environments are not available.
 */
int ml_model_check_nnfw (ml_model_nnfw nnfw, ml_model_hw hw);

/**
 * @}
 */
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TIZEN_NNSTREAMER_SINGLE_H__ */
