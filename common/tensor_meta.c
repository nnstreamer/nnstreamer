/*
 * Gstreamer Tensor Meta
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
 * @file	tensor_meta.c
 * @date	20 June 2018
 * @brief	Meta Data for Tensor type.
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 *
 */

#include <tensor_meta.h>

GType
tensor_meta_api_get_type (void)
{
  static volatile GType type;
  static const gchar *tags[] =
      { "tensor", "other/tensor", "tensors", "other/tensors", NULL };

  if (g_once_init_enter (&type)) {
    GType _type = gst_meta_api_type_register ("TensorMeta", tags);
    g_once_init_leave (&type, _type);
  }

  return type;
}

/**
 * @brief Initialize Tensor Meta Data
 * @param meta Tensor Meta Data
 * @param params Parameters
 * @param buffer GstBuffer
 * @return TRUE/FALSE
 */
static gboolean
tensor_meta_init (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  /* @TODO To be filled */

  TensorMeta *emeta = (TensorMeta *) meta;

  emeta->num_tensors = 0;

  return TRUE;
}

/**
 * @brief Transform Tensor Meta Data
 * @param transbuf GstBuffer to be transformed
 * @param meta Tensor Meta Data
 * @param buffer GstBuffer
 * @param type
 * @param data
 * @return TRUE/FALSE
 */
static gboolean
tensor_meta_transform (GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  /* @TODO To be filled */
  TensorMeta *emeta = (TensorMeta *) meta;
  gst_buffer_add_tensor_meta (transbuf, emeta->num_tensors);

  return TRUE;
}

/**
 * @brief Resource Free for Tensor Meta Data
 * @param meta Tensor Meta Data
 * @param buffer GstBuffer
 * @return TRUE/FALSE
 */
static void
tensor_meta_free (GstMeta * meta, GstBuffer * buffer)
{
  TensorMeta *emeta = (TensorMeta *) meta;
  /* If there is buffer free in here */
  emeta->num_tensors = 0;
}

const GstMetaInfo *
tensor_meta_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;
  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (TENSOR_META_API_TYPE,
        "TensorMeta",
        sizeof (TensorMeta),
        tensor_meta_init,
        tensor_meta_free,
        tensor_meta_transform);
    g_once_init_leave (&meta_info, mi);
  }

  return meta_info;
}

TensorMeta *
gst_buffer_add_tensor_meta (GstBuffer * buffer, gint num_tensors)
{
  TensorMeta *meta;
  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);

  meta = (TensorMeta *) gst_buffer_add_meta (buffer, TENSOR_META_INFO, NULL);

  meta->num_tensors = num_tensors;

  return meta;
}
