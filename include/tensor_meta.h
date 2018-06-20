/*
 * GStreamer Tensor Meta
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	tensor_meta.h
 * @date	20 June 2018
 * @brief	Meta Data for Tensor type.
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 *
 */

#ifndef __GST_TENSOR_META_H__
#define __GST_TENSOR_META_H__

#include <glib.h>
#include <stdint.h>
#include <gst/gst.h>

/**
 * @brief Definition of Tensor Meta Data
 */
typedef struct _TensorMeta {
  GstMeta meta;
  gint num_tensors;
} TensorMeta;

/**
 * @brief Get tensor meta data type. Register Tensor Meta Data API definition
 * @return Tensor Meta Data Type
 */
GType tensor_meta_api_get_type (void);

#define TENSOR_META_API_TYPE (tensor_meta_api_get_type ())

/**
 * @brief Macro to get tensor meta data.
 */
#define gst_buffer_get_tensor_meta(b) \
  ((TensorMeta*) gst_buffer_get_meta ((b), TENSOR_META_API_TYPE))

/**
 * @brief get tensor meta data info
 * @return Tensor Meta Data Info
 */
const GstMetaInfo *tensor_meta_get_info (void);
#define TENSOR_META_INFO (tensor_meta_get_info ())

/**
 * @brief Add tensor meta data
 * @param buffer The buffer to save meta data
 * @param variable to save meta ( number of tensors )
 * @return Tensor Meta Data
 */
TensorMeta * gst_buffer_add_tensor_meta (GstBuffer *buffer,
    gint num_tensors);


#endif /* __GST_TENSOR_META_H__ */
