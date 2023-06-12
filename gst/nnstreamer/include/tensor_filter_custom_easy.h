/**
 * GStreamer Tensor_Filter, Customized Module, Easy Mode
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 *
 */
/**
 * @file	tensor_filter_custom_easy.h
 * @date	24 Oct 2019
 * @brief	Custom tensor processing interface for simple functions
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * How To for NNdevelopers:
 *
 * Case 1. Provide the function as a shared object for other apps.
 * 1. Define struct, "NNStreamer_custom_easy", with the functions defined.
 * 2. Call NNS_custom_easy_register() at a global init/constructor function.
 * 3. Compile as a shared object. (.so in Linux) And install at the custom-filter path.
 * 4. Use NNStreamer (tensor_filter framework=custom_easy, model=${modelname}, custom=${_FILEPATH_OF_YOUR_SO} ...)
 * 5. Note that you may register multiple models (functions) with a single .so.
 *
 * Case 2. Define the function in the app.
 * 1. Define struct, "NNStreamer_custom_easy", with the functions defined.
 * 2. Register the struct with "NNS_custom_easy_register" API.
 * 3. Construct the nnstreamer pipeline and execute it in the app.
 *
 * Note that this does not support flexible dimensions.
 *
 * To Packagers:
 *
 * This file is to be packaged as "devel" package for NN developers.
 */
#ifndef __NNS_TENSOR_FILTER_CUSTOM_EASY_H__
#define __NNS_TENSOR_FILTER_CUSTOM_EASY_H__

#include <stdint.h>
#include "tensor_typedef.h"
#include "tensor_filter_custom.h"

G_BEGIN_DECLS
/**
 * @brief Register the custom-easy tensor function.
 * @param[in] modelname The name of custom-easy tensor function.
 * @param[in] func The tensor function body
 * @param[in/out] private_data The internal data for the function
 * @param[in] in_info Input tensor metadata.
 * @param[in] out_info Output tensor metadata
 * @note NNS_custom_invoke defined in tensor_filter_custom.h
 *       Output buffers for func are preallocated.
 */
extern int NNS_custom_easy_register (const char * modelname,
    NNS_custom_invoke func, void *data,
    const GstTensorsInfo * in_info, const GstTensorsInfo * out_info);

/**
 * @brief Invoke the "main function" with flexible input and output. Output tensor memory should be allocated.
 * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
 * @param[in] info structure of input tensors info
 * @param[out] info structure of output tensors info. The subplugin should fill this info.
 * @param[in] input The array of input tensors. Allocated and filled by tensor_filter/main
 * @param[out] output The array of output tensors. The subplugin should allocate the memory block for output tensor. (data in GstTensorMemory)
 * @note rank limit (NNS_TENSOR_RANK_LIMIT) and typesize (tensor_element_size) defined in tensor_typedef.h
 * @return 0 if success
 */
typedef int (*NNS_custom_invoke_dynamic) (void *private_data, const GstTensorsInfo * in_info, GstTensorsInfo * out_info,
      const GstTensorMemory * input, GstTensorMemory * output);

/**
 * @brief Register the custom-easy tensor function for dynamic invoke.
 * @param[in] modelname The name of custom-easy tensor function.
 * @param[in] func The tensor function body
 * @param[in/out] private_data The internal data for the function
 * @param[in] in_info Input tensor metadata.
 * @note NNS_custom_invoke_dynamic defined in tensor_filter_custom.h
 *       Output buffers should be allocated in the invoke function.
 */
extern int NNS_custom_easy_dynamic_register (const char * modelname,
    NNS_custom_invoke_dynamic func, void *data, const GstTensorsInfo * in_info);

/**
 * @brief Unregister the custom-easy tensor function.
 * @param[in] modelname The registered name of custom-easy tensor function.
 * @return 0 if success, non-zero if error
 */
extern int NNS_custom_easy_unregister (const char * modelname);

G_END_DECLS
#endif /*__NNS_TENSOR_FILTER_CUSTOM_EASY_H__*/
