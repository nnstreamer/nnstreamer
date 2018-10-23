/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 *
 */
/**
 * @file	tensor_filter_custom.h
 * @date	01 Jun 2018
 * @brief	Custom tensor post-processing interface for NNStreamer suite for post-processing code developers.
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * How To for NNdevelopers:
 *
 * 1. Define struct, "NNStreamer_custom", with the functions defined.
 * 2. Compile as a shared object. (.so in Linux)
 * 3. Use NNStreamer (tensor_filter framework=custom, model=FILEPATH_OF_YOUR_SO.so, ...)
 *
 * To Packagers:
 *
 * This file is to be packaged as "devel" package for NN developers.
 */
#ifndef __NNS_TENSOR_FILTER_CUSTOM_H__
#define __NNS_TENSOR_FILTER_CUSTOM_H__

#include <stdint.h>
#include "tensor_typedef.h"

/**
 * @brief A function that is called before calling other functions.
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 * @return The returned pointer will be passed to other functions as "private_data".
 */
typedef void *(*NNS_custom_init_func) (const GstTensorFilterProperties * prop);

/**
 * @brief A function that is called after calling other functions, when it's ready to close.
 * @param[in] private_data If you have allocated *private_data at init, free it here.
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 */
typedef void (*NNS_custom_exit_func) (void *private_data,
    const GstTensorFilterProperties * prop);

/**
 * @brief Get input tensor type.
 * @param[in] private_data The pointer returned by NNStreamer_custom_init.
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 * @param[out] info Structure for tensor info.
 */
typedef int (*NNS_custom_get_input_dimension) (void *private_data,
    const GstTensorFilterProperties * prop, GstTensorsInfo * info);

/**
 * @brief Get output tensor type.
 * @param[in] private_data The pointer returned by NNStreamer_custom_init.
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 * @param[out] info Structure for tensor info.
 */
typedef int (*NNS_custom_get_output_dimension) (void *private_data,
    const GstTensorFilterProperties * prop, GstTensorsInfo * info);

/**
 * @brief Set input dim by framework. Let custom plutin set output dim accordingly.
 * @param[in] private_data The pointer returned by NNStreamer_custom_init
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 * @param[in] in_info Input tensor info designated by the gstreamer framework. Note that this is not a fixed value and gstreamer may try different values during pad-cap negotiations.
 * @param[out] out_info Output tensor info according to the input tensor info.
 *
 * @caution Do not fix internal values based on this call. Gstreamer may call
 * this function repeatedly with different values during pad-cap negotiations.
 * Fix values when invoke is finally called.
 */
typedef int (*NNS_custom_set_input_dimension) (void *private_data,
    const GstTensorFilterProperties * prop, const GstTensorsInfo * in_info, GstTensorsInfo * out_info);

/**
 * @brief Invoke the "main function". Without allocating output buffer. (fill in the given output buffer)
 * @param[in] private_data The pointer returned by NNStreamer_custom_init.
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 * @param[in] input The array of input tensors, each tensor size = dim1 x dim2 x dim3 x dim4 x typesize, allocated by caller
 * @param[out] output The array of output tensors, each tensor size = dim1 x dim2 x dim3 x dim4 x typesize, allocated by caller
 * @return 0 if success
 */
typedef int (*NNS_custom_invoke) (void *private_data,
    const GstTensorFilterProperties * prop, const GstTensorMemory * input, GstTensorMemory * output);

/**
 * @brief Invoke the "main function". Without allocating output buffer. (fill in the given output buffer)
 * @param[in] private_data The pointer returned by NNStreamer_custom_init.
 * @param[in] prop GstTensorFilter's property values. Do not change its values.
 * @param[in] input pointer to input tensor, size = dim1 x dim2 x dim3 x dim4 x typesize, allocated by caller
 * @param[out] output The array of output tensors, each tensor size = dim1 x dim2 x dim3 x dim4 x typesize, the memory block for output tensor should be allocated. (data in GstTensorMemory)
 * @param[out] size The allocated size.
 * @return The output buffer allocated in the invoke function
 */
typedef int (*NNS_custom_allocate_invoke) (void *private_data,
    const GstTensorFilterProperties * prop, const GstTensorMemory * input, GstTensorMemory * output);

/**
 * @brief Custom Filter Class
 *
 * Note that exery function pointer is MANDATORY!
 */
struct _NNStreamer_custom_class
{
  NNS_custom_init_func initfunc; /**< called before any other callbacks from tensor_filter_custom.c */
  NNS_custom_exit_func exitfunc; /**< will not call other callbacks after this call */
  NNS_custom_get_input_dimension getInputDim; /**< a custom filter is required to provide input tensor dimension unless setInputdim is defined. */
  NNS_custom_get_output_dimension getOutputDim; /**< a custom filter is require dto provide output tensor dimension unless setInputDim is defined. */
  NNS_custom_set_input_dimension setInputDim; /**< without getI/O-Dim, this allows framework to set input dimension and get output dimension from the custom filter according to the input dimension */
  NNS_custom_invoke invoke; /**< the main function, "invoke", that transforms input to output. invoke is supposed to fill in the given output buffer. (invoke) XOR (allocate_invoke) MUST hold. */
  NNS_custom_allocate_invoke allocate_invoke; /**< the main function, "allocate & invoke", that transforms input to output. allocate_invoke is supposed to allocate output buffer by itself. (invoke) XOR (allocate_invoke) MUST hold. */
};
typedef struct _NNStreamer_custom_class NNStreamer_custom_class;

/**
 * @brief A custom filter MUST define NNStreamer_custom. This object represents the custom filter itself.
 */
extern NNStreamer_custom_class *NNStreamer_custom;

#endif /*__NNS_TENSOR_FILTER_CUSTOM_H__*/
