/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
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
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	tensor_filter_custom.h
 * @date	01 Jun 2018
 * @brief	Custom tensor post-processing interface for NNStreamer suite for post-processing code developers.
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
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

/**
 * @brief A function that is called before calling other functions.
 * @return The returned pointer will be passed to other functions as "private_data".
 */
typedef void *(*NNS_custom_init_func)(void);

/**
 * @brief A function that is called after calling other functions, when it's ready to close.
 * @param[in] private_data If you have allocated *private_data at init, free it here.
 */
typedef void (*NNS_custom_exit_func)(void *private_data);

/**
 * @brief Get input tensor type.
 * @param[in] private_data The pointer returned by NNStreamer_custom_exit.
 * @param[out] inputDimension uint32_t[NNS_TENSOR_RANK_LIMIT]
 * @param[out] type Type of each element in the input tensor
 */
typedef int (*NNS_custom_get_input_dimension)(void *private_data,
    uint32_t *inputDimension, tensor_type *type);

/**
 * @brief Get output tensor type.
 * @param[in] private_data The pointer returned by NNStreamer_custom_exit.
 * @param[out] outputDimension uint32_t[NNS_TENSOR_RANK_LIMIT]
 * @param[out] type Type of each element in the output tensor
 */
typedef int (*NNS_custom_get_output_dimension)(void *private_data,
    uint32_t *outputDimension, tensor_type *type);

/**
 * @brief Invoke the "main function".
 * @param[in] private_data The pointer returned by NNStreamer_custom_exit.
 * @param[in] inputPtr pointer to input tensor, size = dim1 x dim2 x dim3 x dim4 x typesize, allocated by caller
 * @param[in] inputPtr pointer to output tensor, size = dim1 x dim2 x dim3 x dim4 x typesize, allocated by caller
 */
typedef int (*NNS_custom_invoke)(void *private_data,
    void *inputPtr, void *outputPtr);

/**
 * @brief Custom Filter Class
 *
 * Note that exery function pointer is MANDATORY!
 */
struct _NNStreamer_custom_class {
  NNS_custom_init_func initfunc;
  NNS_custom_exit_func exitfunc;
  NNS_custom_get_input_dimension getInputDim;
  NNS_custom_get_output_dimension getOutputDim;
  NNS_custom_invoke invoke;
};
typedef struct _NNStreamer_custom_class NNStreamer_custom_class;
extern NNStreamer_custom_class *NNStreamer_custom;

#endif /*__NNS_TENSOR_FILTER_CUSTOM_H__*/
