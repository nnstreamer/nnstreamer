/*
 * NNStreamer Common Header
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
 * @file	tensor_common.h
 * @date	23 May 2018
 * @brief	Common header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 */

#ifndef __GST_TENSOR_COMMON_H__
#define __GST_TENSOR_COMMON_H__

#include <glib.h>

G_BEGIN_DECLS

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
} nns_tensor_type;

/**
 * @brief Possible input stream types for other/tensor.
 *
 * This is realted with media input stream to other/tensor.
 * There is no restrictions for the outputs.
 */
typedef enum _nns_media_type {
  _NNS_VIDEO = 0,
  _NNS_AUDIO, /* Not Supported Yet */
  _NNS_STRING, /* Not Supported Yet */

  _NNS_MEDIA_END,
} nns_media_type;

/**
 * @brief Byte-per-element of each tensor element type.
 */
const unsigned int nns_tensor_element_size[] = {
        [_NNS_INT32] = 4,
        [_NNS_UINT32] = 4,
        [_NNS_INT16] = 2,
        [_NNS_UINT16] = 2,
        [_NNS_INT8] = 1,
        [_NNS_UINT8] = 1,
        [_NNS_FLOAT64] = 8,
        [_NNS_FLOAT32] = 4,
};

/**
 * @brief String representations for each tensor element type.
 */
const gchar* nns_tensor_element_typename[] = {
        [_NNS_INT32] = "int32",
        [_NNS_UINT32] = "uint32",
        [_NNS_INT16] = "int16",
        [_NNS_UINT16] = "uint16",
        [_NNS_INT8] = "int8",
        [_NNS_UINT8] = "uint8",
        [_NNS_FLOAT64] = "float64",
        [_NNS_FLOAT32] = "float32",
};

G_END_DECLS

#endif /* __GST_TENSOR_COMMON_H__ */
