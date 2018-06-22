/**
 * NNStreamer Common Header, Typedef part, for export as devel package.
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
 */
/**
 * @file	tensor_common_typedef.h
 * @date	01 Jun 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * To Packagers:
 *
 * This fils it to be packaged as "devel" package for NN developers.
 */
#ifndef __GST_TENSOR_TYPEDEF_H__
#define __GST_TENSOR_TYPEDEF_H__

#define NNS_TENSOR_RANK_LIMIT	(4)
/**
 * @brief Possible data element types of other/tensor.
 *
 * The current version supports NNS_UINT8 only as video-input.
 * There is no restrictions for inter-NN or sink-to-app.
 */
typedef enum _nns_tensor_type {
  _NNS_INT32 = 0,
  _NNS_UINT32,
  _NNS_INT16,
  _NNS_UINT16,
  _NNS_INT8,
  _NNS_UINT8,
  _NNS_FLOAT64,
  _NNS_FLOAT32,

  _NNS_END,
} tensor_type;

/**
 * @brief NN Frameworks available for the tensor_filter element.
 */
typedef enum _nnfw_type {
  _T_F_UNDEFINED = 0, /**< Not defined or supported. Cannot proceed in this status */

  _T_F_CUSTOM, /**< Custom filter provided as a shared object (dysym) */
  _T_F_TENSORFLOW_LITE, /**< In Progress */
  _T_F_TENSORFLOW, /**< NYI */
  _T_F_CAFFE2, /**< NYI */

  _T_F_NNFW_END,
} nnfw_type;

struct _GstTensor_Filter_Framework;
typedef struct _GstTensor_Filter_Framework GstTensor_Filter_Framework;

/**
 * @brief Tensor_Filter's properties (internal data structure)
 *
 * Because custom filters of tensor_filter may need to access internal data
 * of Tensor_Filter, we define this data structure here.
 */
typedef struct _GstTensor_Filter_Properties
{
  gboolean silent; /**< Verbose mode if FALSE */
  gboolean inputConfigured; /**< TRUE if input dimension is configured */
  gboolean outputConfigured; /** < TRUE if output dimension is configured */
  nnfw_type nnfw; /**< The enum value of corresponding NNFW. _T_F_UNDEFINED if not configured */
  GstTensor_Filter_Framework *fw; /**< The implementation core of the NNFW. NULL if not configured */
  const gchar *modelFilename; /**< Filepath to the model file (as an argument for NNFW) */

  uint32_t inputDimension[NNS_TENSOR_RANK_LIMIT]; /**< The input tensor dimension */
  tensor_type inputType; /**< The type for each element in the input tensor */
  gboolean inputCapNegotiated;
  uint32_t outputDimension[NNS_TENSOR_RANK_LIMIT]; /**< The output tensor dimension */
  tensor_type outputType; /**< The type for each element in the output tensor */
  gboolean outputCapNegotiated;

  const gchar *customProperties; /**< sub-plugin specific custom property values in string */
} GstTensor_Filter_Properties;

typedef uint32_t tensor_dim[NNS_TENSOR_RANK_LIMIT];

#endif /*__GST_TENSOR_TYPEDEF_H__*/
