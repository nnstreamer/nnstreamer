/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_server.c
 * @date    03 Aug 2021
 * @brief   GStreamer plugin to handle meta_query for server elements
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tensor_query_server.h"
#include <nnstreamer_util.h>
#include <tensor_typedef.h>
#include <tensor_common.h>

/**
 * @brief mutex for tensor-query server table.
 */
G_LOCK_DEFINE_STATIC (query_server_table);

/**
 * @brief Table for query server data.
 */
static GHashTable *_qs_table = NULL;

static void init_queryserver (void) __attribute__((constructor));
static void fini_queryserver (void) __attribute__((destructor));

/**
 * @brief Getter to get nth GstTensorQueryServerInfo.
 */
query_server_info_handle
gst_tensor_query_server_get_data (guint id)
{
  gpointer p;

  G_LOCK (query_server_table);
  p = g_hash_table_lookup (_qs_table, GUINT_TO_POINTER (id));
  G_UNLOCK (query_server_table);

  return p;
}

/**
 * @brief Add GstTensorQueryServerInfo into hash table.
 */
query_server_info_handle
gst_tensor_query_server_add_data (guint id)
{
  GstTensorQueryServerInfo *data = NULL;

  data = (GstTensorQueryServerInfo *) gst_tensor_query_server_get_data (id);

  if (NULL != data) {
    return data;
  }

  data = g_try_new0 (GstTensorQueryServerInfo, 1);
  if (NULL == data) {
    GST_ERROR ("Failed to allocate memory for tensor query server data.");
    return NULL;
  }

  g_mutex_init (&data->lock);
  g_cond_init (&data->cond);
  data->id = id;
  data->sink_host = NULL;
  data->sink_port = 0;
  data->configured = FALSE;
  data->sink_caps_str = NULL;

  G_LOCK (query_server_table);
  g_hash_table_insert (_qs_table, GUINT_TO_POINTER (id), data);
  G_UNLOCK (query_server_table);

  return data;
}

/**
 * @brief Remove GstTensorQueryServerInfo.
 */
void
gst_tensor_query_server_remove_data (query_server_info_handle server_info_h)
{
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;

  if (NULL == data) {
    return;
  }

  G_LOCK (query_server_table);
  g_hash_table_remove (_qs_table, GUINT_TO_POINTER (data->id));
  G_UNLOCK (query_server_table);
  g_free (data->sink_host);
  data->sink_host = NULL;
  g_free (data->sink_caps_str);
  data->sink_caps_str = NULL;
  g_cond_clear (&data->cond);
  g_mutex_clear (&data->lock);
  g_free (data);
}

/**
 * @brief Wait until the sink is configured and get server info handle.
 */
gboolean
gst_tensor_query_server_wait_sink (query_server_info_handle server_info_h)
{
  gint64 end_time;
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;

  if (NULL == data) {
    return FALSE;
  }

  end_time = g_get_monotonic_time () +
      DEFAULT_QUERY_INFO_TIMEOUT * G_TIME_SPAN_SECOND;
  g_mutex_lock (&data->lock);
  while (!data->configured) {
    if (!g_cond_wait_until (&data->cond, &data->lock, end_time)) {
      g_mutex_unlock (&data->lock);
      g_critical ("Failed to get server sink info.");
      return FALSE;
    }
  }
  g_mutex_unlock (&data->lock);

  return TRUE;
}

/**
 * @brief set sink caps string.
 */
void
gst_tensor_query_server_set_sink_caps_str (query_server_info_handle
    server_info_h, const gchar * caps_str)
{
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;

  if (NULL == data) {
    return;
  }
  g_mutex_lock (&data->lock);
  if (caps_str) {
    g_free (data->sink_caps_str);
    data->sink_caps_str = g_strdup (caps_str);
  }
  data->configured = TRUE;
  g_cond_broadcast (&data->cond);
  g_mutex_unlock (&data->lock);
}

/**
 * @brief get sink caps string.
 */
gchar *
gst_tensor_query_server_get_sink_caps_str (query_server_info_handle
    server_info_h)
{
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;
  gchar *caps_str = NULL;

  if (NULL == data) {
    return caps_str;
  }

  g_mutex_lock (&data->lock);
  caps_str = g_strdup (data->sink_caps_str);
  g_mutex_unlock (&data->lock);

  return caps_str;
}

/**
 * @brief set sink host
 */
void
gst_tensor_query_server_set_sink_host (query_server_info_handle server_info_h,
    gchar * host, guint16 port)
{
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;

  if (NULL == data) {
    return;
  }

  g_mutex_lock (&data->lock);
  data->sink_host = g_strdup (host);
  data->sink_port = port;
  g_mutex_unlock (&data->lock);
}

/**
 * @brief get sink host
 */
gchar *
gst_tensor_query_server_get_sink_host (query_server_info_handle server_info_h)
{
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;
  gchar *sink_host = NULL;

  if (NULL == data) {
    return NULL;
  }

  g_mutex_lock (&data->lock);
  sink_host = g_strdup (data->sink_host);
  g_mutex_unlock (&data->lock);

  return sink_host;
}

/**
 * @brief get sink port
 */
guint16
gst_tensor_query_server_get_sink_port (query_server_info_handle server_info_h)
{
  GstTensorQueryServerInfo *data = (GstTensorQueryServerInfo *) server_info_h;
  guint16 sink_port = 0;

  if (NULL == data) {
    return sink_port;
  }

  g_mutex_lock (&data->lock);
  sink_port = data->sink_port;
  g_mutex_unlock (&data->lock);

  return sink_port;
}

/**
 * @brief Initialize the query server.
 */
static void
init_queryserver (void)
{
  G_LOCK (query_server_table);
  g_assert (NULL == _qs_table); /** Internal error (duplicated init call?) */
  _qs_table = g_hash_table_new (g_direct_hash, g_direct_equal);
  G_UNLOCK (query_server_table);
}

/**
 * @brief Destruct the query server.
 */
static void
fini_queryserver (void)
{
  G_LOCK (query_server_table);
  g_assert (_qs_table); /** Internal error (init not called?) */
  g_hash_table_destroy (_qs_table);
  _qs_table = NULL;
  G_UNLOCK (query_server_table);
}
