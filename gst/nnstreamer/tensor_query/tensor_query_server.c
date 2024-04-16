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
 * @brief Internal function to release query server data.
 */
static void
_release_server_data (gpointer data)
{
  GstTensorQueryServer *_data = (GstTensorQueryServer *) data;

  if (!_data)
    return;

  g_mutex_lock (&_data->lock);
  if (_data->edge_h) {
    nns_edge_release_handle (_data->edge_h);
    _data->edge_h = NULL;
  }
  g_mutex_unlock (&_data->lock);

  g_mutex_clear (&_data->lock);
  g_cond_clear (&_data->cond);

  g_free (_data);
}

/**
 * @brief Get nnstreamer edge server handle.
 */
static GstTensorQueryServer *
gst_tensor_query_server_get_handle (const guint id)
{
  GstTensorQueryServer *data;

  G_LOCK (query_server_table);
  data = g_hash_table_lookup (_qs_table, GUINT_TO_POINTER (id));
  G_UNLOCK (query_server_table);

  return data;
}

/**
 * @brief Add nnstreamer edge server handle into hash table.
 */
gboolean
gst_tensor_query_server_add_data (const guint id)
{
  GstTensorQueryServer *data;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL != data) {
    return TRUE;
  }

  data = g_try_new0 (GstTensorQueryServer, 1);
  if (NULL == data) {
    nns_loge ("Failed to allocate memory for tensor query server data.");
    return FALSE;
  }

  g_mutex_init (&data->lock);
  g_cond_init (&data->cond);
  data->id = id;
  data->configured = FALSE;

  G_LOCK (query_server_table);
  g_hash_table_insert (_qs_table, GUINT_TO_POINTER (id), data);
  G_UNLOCK (query_server_table);

  return TRUE;
}

/**
 * @brief Prepare edge connection and its handle.
 */
gboolean
gst_tensor_query_server_prepare (const guint id,
    nns_edge_connect_type_e connect_type, GstTensorQueryEdgeInfo * edge_info)
{
  GstTensorQueryServer *data;
  gchar *port_str, *id_str;
  gboolean prepared = FALSE;
  gint ret;

  data = gst_tensor_query_server_get_handle (id);
  if (NULL == data) {
    return FALSE;
  }

  g_mutex_lock (&data->lock);
  if (data->edge_h == NULL) {
    id_str = g_strdup_printf ("%u", id);

    ret = nns_edge_create_handle (id_str, connect_type,
        NNS_EDGE_NODE_TYPE_QUERY_SERVER, &data->edge_h);
    g_free (id_str);

    if (NNS_EDGE_ERROR_NONE != ret) {
      GST_ERROR ("Failed to get nnstreamer edge handle.");
      goto done;
    }
  }

  if (edge_info) {
    if (edge_info->host) {
      nns_edge_set_info (data->edge_h, "HOST", edge_info->host);
    }
    if (edge_info->port > 0) {
      port_str = g_strdup_printf ("%u", edge_info->port);
      nns_edge_set_info (data->edge_h, "PORT", port_str);
      g_free (port_str);
    }
    if (edge_info->dest_host) {
      nns_edge_set_info (data->edge_h, "DEST_HOST", edge_info->dest_host);
    }
    if (edge_info->dest_port > 0) {
      port_str = g_strdup_printf ("%u", edge_info->dest_port);
      nns_edge_set_info (data->edge_h, "DEST_PORT", port_str);
      g_free (port_str);
    }
    if (edge_info->topic) {
      nns_edge_set_info (data->edge_h, "TOPIC", edge_info->topic);
    }

    nns_edge_set_event_callback (data->edge_h, edge_info->cb, edge_info->pdata);

    ret = nns_edge_start (data->edge_h);
    if (NNS_EDGE_ERROR_NONE != ret) {
      nns_loge
          ("Failed to start NNStreamer-edge. Please check server IP and port.");
      goto done;
    }
  }

  prepared = TRUE;

done:
  g_mutex_unlock (&data->lock);
  return prepared;
}

/**
 * @brief Send buffer to connected edge device.
 */
gboolean
gst_tensor_query_server_send_buffer (const guint id, GstBuffer * buffer)
{
  GstTensorQueryServer *data;
  GstMetaQuery *meta_query;
  nns_edge_data_h data_h;
  guint i, num_tensors = 0;
  gint ret = NNS_EDGE_ERROR_NONE;
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo map[NNS_TENSOR_SIZE_LIMIT];
  gchar *val;
  gboolean sent = FALSE;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL == data) {
    nns_loge ("Failed to send buffer, server handle is null.");
    return FALSE;
  }

  meta_query = gst_buffer_get_meta_query (buffer);
  if (!meta_query) {
    nns_loge ("Failed to send buffer, cannot get tensor query meta.");
    return FALSE;
  }

  ret = nns_edge_data_create (&data_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to create edge data handle in query server.");
    return FALSE;
  }

  num_tensors = gst_tensor_buffer_get_count (buffer);
  for (i = 0; i < num_tensors; i++) {
    mem[i] = gst_tensor_buffer_get_nth_memory (buffer, i);

    if (!gst_memory_map (mem[i], &map[i], GST_MAP_READ)) {
      ml_loge ("Cannot map the %uth memory in gst-buffer.", i);
      gst_memory_unref (mem[i]);
      num_tensors = i;
      goto done;
    }

    nns_edge_data_add (data_h, map[i].data, map[i].size, NULL);
  }

  val = g_strdup_printf ("%lld", (long long) meta_query->client_id);
  nns_edge_data_set_info (data_h, "client_id", val);
  g_free (val);

  g_mutex_lock (&data->lock);
  ret = nns_edge_send (data->edge_h, data_h);
  g_mutex_unlock (&data->lock);

  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to send edge data handle in query server.");
    goto done;
  }

  sent = TRUE;

done:
  for (i = 0; i < num_tensors; i++) {
    gst_memory_unmap (mem[i], &map[i]);
    gst_memory_unref (mem[i]);
  }

  nns_edge_data_destroy (data_h);

  return sent;
}

/**
 * @brief Release nnstreamer edge handle of query server.
 */
void
gst_tensor_query_server_release_edge_handle (const guint id)
{
  GstTensorQueryServer *data;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL == data) {
    return;
  }

  g_mutex_lock (&data->lock);
  if (data->edge_h) {
    nns_edge_release_handle (data->edge_h);
    data->edge_h = NULL;
  }
  g_mutex_unlock (&data->lock);
}

/**
 * @brief Remove GstTensorQueryServer.
 */
void
gst_tensor_query_server_remove_data (const guint id)
{
  G_LOCK (query_server_table);
  if (g_hash_table_lookup (_qs_table, GUINT_TO_POINTER (id)))
    g_hash_table_remove (_qs_table, GUINT_TO_POINTER (id));
  G_UNLOCK (query_server_table);
}

/**
 * @brief Wait until the sink is configured and get server info handle.
 */
gboolean
gst_tensor_query_server_wait_sink (const guint id)
{
  gint64 end_time;
  GstTensorQueryServer *data;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL == data) {
    return FALSE;
  }

  end_time = g_get_monotonic_time () +
      DEFAULT_QUERY_INFO_TIMEOUT * G_TIME_SPAN_SECOND;
  g_mutex_lock (&data->lock);
  while (!data->configured) {
    if (!g_cond_wait_until (&data->cond, &data->lock, end_time)) {
      g_mutex_unlock (&data->lock);
      ml_loge ("Failed to get server sink info.");
      return FALSE;
    }
  }
  g_mutex_unlock (&data->lock);

  return TRUE;
}

/**
 * @brief set query server sink configured.
 */
void
gst_tensor_query_server_set_configured (const guint id)
{
  GstTensorQueryServer *data;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL == data) {
    return;
  }

  g_mutex_lock (&data->lock);
  data->configured = TRUE;
  g_cond_broadcast (&data->cond);
  g_mutex_unlock (&data->lock);
}

/**
 * @brief set query server caps.
 */
void
gst_tensor_query_server_set_caps (const guint id, const gchar * caps_str)
{
  GstTensorQueryServer *data;
  gchar *prev_caps_str, *new_caps_str;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL == data) {
    return;
  }

  g_mutex_lock (&data->lock);

  prev_caps_str = new_caps_str = NULL;
  nns_edge_get_info (data->edge_h, "CAPS", &prev_caps_str);
  if (!prev_caps_str)
    prev_caps_str = g_strdup ("");
  new_caps_str = g_strdup_printf ("%s%s", prev_caps_str, caps_str);
  nns_edge_set_info (data->edge_h, "CAPS", new_caps_str);

  g_free (prev_caps_str);
  g_free (new_caps_str);

  g_mutex_unlock (&data->lock);
}

/**
 * @brief Initialize the query server.
 */
static void
init_queryserver (void)
{
  G_LOCK (query_server_table);
  g_assert (NULL == _qs_table); /** Internal error (duplicated init call?) */
  _qs_table = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL,
      _release_server_data);
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
