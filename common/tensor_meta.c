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
#include <string.h>

GType
gst_meta_tensor_api_get_type (void)
{
  static volatile GType type;
  static const gchar *tags[] = { "tensor", "tensors", NULL };

  if (g_once_init_enter (&type)) {
    GType _type;
    const GstMetaInfo *meta_info = gst_meta_get_info ("GstMetaTensor");

    if (meta_info) {
      _type = meta_info->api;
    } else {
      _type = gst_meta_api_type_register ("GstMetaTensorAPI", tags);
    }
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
gst_meta_tensor_init (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  /* @TODO To be filled */

  GstMetaTensor *emeta = (GstMetaTensor *) meta;

  emeta->num_tensors = 0;
  emeta->dimensions = NULL;

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
gst_meta_tensor_transform (GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  /* @TODO To be filled */
  GstMetaTensor *dest_meta = GST_META_TENSOR_ADD (transbuf);
  GstMetaTensor *src_meta = (GstMetaTensor *) meta;

  dest_meta->num_tensors = src_meta->num_tensors;

  return TRUE;
}

/**
 * @brief Resource Free for Tensor Meta Data
 * @param meta Tensor Meta Data
 * @param buffer GstBuffer
 * @return TRUE/FALSE
 */
static void
gst_meta_tensor_free (GstMeta * meta, GstBuffer * buffer)
{
  GstMetaTensor *emeta = (GstMetaTensor *) meta;
  /* If there is buffer free in here */
  emeta->num_tensors = 0;
}

/**
 * @brief Get Gst Tensor Meta Info
 * @return Gst Tensor Meta Info
 */
const GstMetaInfo *
gst_meta_tensor_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;
  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (GST_META_TENSOR_API_TYPE,
        "GstMetaTensor",
        sizeof (GstMetaTensor),
        (GstMetaInitFunction) gst_meta_tensor_init,
        (GstMetaFreeFunction) gst_meta_tensor_free,
        (GstMetaTransformFunction) gst_meta_tensor_transform);
    g_once_init_leave (&meta_info, mi);
  }

  return meta_info;
}

GstMetaTensor *
gst_buffer_add_meta_tensor (GstBuffer * buffer)
{
  GstMetaTensor *meta;
  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);

  meta =
      (GstMetaTensor *) gst_buffer_add_meta (buffer, GST_META_TENSOR_INFO,
      NULL);

  return meta;
}

GstMetaTensor *
gst_make_tensors (GstBuffer * buffer)
{
  return GST_META_TENSOR_ADD (buffer);
}

GstMetaTensor *
gst_append_tensor (GstBuffer * buffer, GstMemory * mem, tensor_dim * dim)
{
  tensor_dim *d;
  GstMapInfo dest_info;
  gst_buffer_map (buffer, &dest_info, GST_MAP_WRITE);

  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);

  gst_buffer_append_memory (buffer, mem);
  GstMetaTensor *meta = GST_META_TENSOR_GET (buffer);
  if (!meta) {
    meta = gst_make_tensors (buffer);
  }

  meta->num_tensors = meta->num_tensors + 1;

  d = g_slice_new (tensor_dim);
  memcpy (d, dim, sizeof (tensor_dim));
  meta->dimensions = g_list_append (meta->dimensions, d);

  return meta;
}

GstMemory *
gst_get_tensor (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta;
  GstMemory *mem;
  meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (!meta) {
    return (GstMemory *) buffer;
  } else {
    mem = gst_buffer_get_memory (buffer, nth);
    return mem;
  }
}

tensor_dim *
gst_get_tensordim (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta;
  tensor_dim *dim;
  meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (meta) {
    dim = (tensor_dim *) g_list_nth_data (meta->dimensions, nth);
    return dim;
  } else {
    return NULL;
  }
}

GstFlowReturn
gst_remove_tensor (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (meta) {
    if (meta->num_tensors == 0)
      return GST_FLOW_ERROR;
    meta->num_tensors = meta->num_tensors - 1;
    GList *list = meta->dimensions;
    gint th = 0;
    while (list != NULL) {
      GList *next = list->next;
      if (th == nth) {
        meta->dimensions = g_list_delete_link (meta->dimensions, list);
      }
      th++;
      list = next;
    }
    gst_buffer_remove_memory (buffer, nth);
  }

  return GST_FLOW_OK;
}

gint
gst_get_num_tensors (GstBuffer * buffer)
{
  GstMetaTensor *meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  return meta->num_tensors;
}
