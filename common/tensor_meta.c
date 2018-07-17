/**
 * Gstreamer Tensor Meta
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	tensor_meta.c
 * @date	20 June 2018
 * @brief	Meta Data for Tensor type.
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <tensor_meta.h>
#include <string.h>
#include <stdlib.h>
#include <glib.h>
#include <dlfcn.h>

/**
 * @brief return meta type (GstMetaTensor)
 * @return GType GstMetaTensor
 */
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
  GstMetaTensor *emeta = (GstMetaTensor *) meta;

  emeta->num_tensors = 0;
  emeta->dimensions = NULL;
  emeta->types = NULL;
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

  GList *l = emeta->dimensions;
  while (l != NULL) {
    GList *next = l->next;
    g_free (next->data);
  }
  g_list_free (emeta->dimensions);

  l = emeta->types;
  while (l != NULL) {
    GList *next = l->next;
    g_free (next->data);
  }
  g_list_free (emeta->types);

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

/**
 * @brief Add Meta into buffer
 */
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

/**
 * @brief fill this in
 */
GstMetaTensor *
gst_make_tensors (GstBuffer * buffer)
{
  return GST_META_TENSOR_ADD (buffer);
}

/**
 * @brief append tensor into buffer
 */
GstMetaTensor *
gst_append_tensor (GstBuffer * buffer, GstMemory * mem, tensor_dim dim,
    tensor_type type)
{
  tensor_dim *d;
  tensor_type *t;
  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);

  GstMetaTensor *meta = GST_META_TENSOR_GET (buffer);
  if (!meta) {
    meta = gst_make_tensors (buffer);
  }

  gst_buffer_append_memory (buffer, mem);

  meta->num_tensors = meta->num_tensors + 1;
  if (gst_buffer_n_memory (buffer) != meta->num_tensors)
    err_print
        ("Number of memory block in buffer(%d) is not compatible with Meta (%d)\n",
        gst_buffer_n_memory (buffer), meta->num_tensors);
  d = g_slice_new (tensor_dim);
  memcpy (d, dim, sizeof (tensor_dim));
  meta->dimensions = g_list_append (meta->dimensions, d);

  t = g_slice_new (tensor_type);
  *t = type;
  meta->types = g_list_append (meta->types, t);

  return meta;
}

/**
 * @brief Get tensor GstMemory
 */
GstMemory *
gst_get_tensor (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta;
  GstMemory *mem;
  meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (!meta) {
    err_print ("Cannot get meta!\n");
    return (GstMemory *) buffer;
  } else {
    if (gst_buffer_n_memory (buffer) != gst_get_num_tensors (buffer))
      err_print
          ("Number of memory block in buffer(%d) is not compatible with Meta (%d)\n",
          gst_buffer_n_memory (buffer), gst_get_num_tensors (buffer));
    mem = gst_buffer_get_memory (buffer, nth);
    gst_memory_unref (mem);
    return mem;
  }
}

/**
 * @brief Get tensor dimension
 */
tensor_dim *
gst_get_tensordim (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta;
  tensor_dim *dim = NULL;
  meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (meta) {
    dim = g_list_nth_data (meta->dimensions, nth);
  }
  return dim;
}

/**
 * @brief Get tensor type
 */
tensor_type
gst_get_tensortype (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta;
  tensor_type *t;
  tensor_type type;
  meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (meta) {
    t = g_list_nth_data (meta->types, nth);
    type = *t;
    return type;
  } else {
    return _NNS_END;
  }
}

/**
 * @brief Remove tensor in tensors
 */
GstFlowReturn
gst_remove_tensor (GstBuffer * buffer, gint nth)
{
  GstMetaTensor *meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  if (meta) {
    if (meta->num_tensors == 0) {
      return GST_FLOW_ERROR;
    }

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

    list = meta->types;
    th = 0;
    while (list != NULL) {
      GList *next = list->next;
      if (th == nth) {
        meta->types = g_list_delete_link (meta->types, list);
      }
      th++;
      list = next;
    }
    gst_buffer_remove_memory (buffer, nth);
  }

  return GST_FLOW_OK;
}

/**
 * @brief @todo fill this in
 */
gint
gst_get_num_tensors (GstBuffer * buffer)
{
  GstMetaTensor *meta = (GstMetaTensor *) gst_buffer_get_meta_tensor (buffer);
  return meta->num_tensors;
}

/**
 * @brief @todo fill this in
 */
GArray *
parse_dimensions (const gchar * dim_string)
{
  GArray *dimensions;
  gint i, num_tensors;
  gchar **arr;
  arr = g_strsplit_set (dim_string, ",.;/", -1);
  num_tensors = g_strv_length (arr);

  dimensions =
      g_array_sized_new (FALSE, FALSE, sizeof (tensor_dim *), num_tensors);
  for (i = 0; i < num_tensors; i++) {
    gchar **p;
    gint num, k;
    tensor_dim *d;

    p = g_strsplit_set (arr[i], ":", -1);
    num = g_strv_length (p);

    d = g_new0 (tensor_dim, 1);
    for (k = 0; k < num; k++) {
      (*d)[k] = atoi (p[k]);
    }

    g_array_append_val (dimensions, d);
    g_strfreev (p);
  }

  g_strfreev (arr);

  return dimensions;
}

/**
 * @brief @todo fill this in
 */
GArray *
parse_types (const gchar * types_string)
{
  gchar **charbuf;
  gint num_type, i;
  GArray *types;
  charbuf = g_strsplit_set (types_string, ",.:/", -1);
  num_type = g_strv_length (charbuf);

  types = g_array_sized_new (FALSE, FALSE, sizeof (tensor_type *), num_type);

  for (i = 0; i < num_type; i++) {
    tensor_type *t = g_new0 (tensor_type, 1);
    (*t) = get_tensor_type (charbuf[i]);
    g_array_append_val (types, t);
  }

  g_strfreev (charbuf);
  return types;
}
