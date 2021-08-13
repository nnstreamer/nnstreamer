/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Junhwan Kim <jejudo.kim@samsung.com>
 *
 * @file    tensor_meta.c
 * @date    09 Aug 2021
 * @brief   Internal tensor meta implementation for nnstreamer
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <nnstreamer_util.h>
#include "tensor_meta.h"

/**
 * @brief Define meta_query type to register
 */
GType
gst_meta_query_api_get_type (void)
{
  static GType type = 0;
  static const gchar *tags[] = {
    NULL
  };

  if (g_once_init_enter (&type)) {
    GType _type;
    const GstMetaInfo *meta_info = gst_meta_get_info ("GstMetaQuery");
    if (meta_info) {
      _type = meta_info->api;
    } else {
      _type = gst_meta_api_type_register ("GstMetaQueryAPI", tags);
    }
    g_once_init_leave (&type, _type);
  }
  return type;
}

/**
 * @brief meta_query init
 */
static gboolean
gst_meta_query_init (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  GstMetaQuery *emeta = (GstMetaQuery *) meta;
  UNUSED (params);
  UNUSED (buffer);
  emeta->client_id = 0;
  return TRUE;
}

/**
 * @brief free meta_query
 */
static void
gst_meta_query_free (GstMeta * meta, GstBuffer * buffer)
{
  UNUSED (meta);
  UNUSED (buffer);
}

/**
 * @brief tensor_query meta data transform (source to dest)
 */
static gboolean
gst_meta_query_transform (GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  GstMetaQuery *dest_meta = gst_buffer_add_meta_query (transbuf);
  GstMetaQuery *src_meta = (GstMetaQuery *) meta;
  UNUSED (buffer);
  UNUSED (type);
  UNUSED (data);
  dest_meta->client_id = src_meta->client_id;
  return TRUE;
}

/**
 * @brief Get meta_query info
 */
const GstMetaInfo *
gst_meta_query_get_info (void)
{
  static const GstMetaInfo *meta_query_info = NULL;

  if (g_once_init_enter (&meta_query_info)) {
    const GstMetaInfo *meta = gst_meta_register (GST_META_QUERY_API_TYPE,
        "GstMetaQuery", sizeof *meta,
        gst_meta_query_init,
        gst_meta_query_free,
        gst_meta_query_transform);
    g_once_init_leave (&meta_query_info, meta);
  }
  return meta_query_info;
}
