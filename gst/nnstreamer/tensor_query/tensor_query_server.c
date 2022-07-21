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
#include "nnstreamer-edge.h"

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
 * @brief Get nnstreamer edge server handle.
 */
edge_server_handle
gst_tensor_query_server_get_handle (char *id)
{
  edge_server_handle p;

  G_LOCK (query_server_table);
  p = g_hash_table_lookup (_qs_table, id);
  G_UNLOCK (query_server_table);

  return p;
}

/**
 * @brief Add nnstreamer edge server handle into hash table.
 */
edge_server_handle
gst_tensor_query_server_add_data (char *id)
{
  GstTensorQueryServer *data = NULL;
  int ret;

  data = gst_tensor_query_server_get_handle (id);

  if (NULL != data) {
    return data;
  }

  data = g_try_new0 (GstTensorQueryServer, 1);
  if (NULL == data) {
    GST_ERROR ("Failed to allocate memory for tensor query server data.");
    return NULL;
  }

  g_mutex_init (&data->lock);
  g_cond_init (&data->cond);
  data->id = id;
  data->configured = FALSE;

  ret = nns_edge_create_handle (id, "TEMP_SERVER_TOPIC", &data->edge_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    GST_ERROR ("Failed to get nnstreamer edge handle.");
    gst_tensor_query_server_remove_data (data);
    return NULL;
  }

  G_LOCK (query_server_table);
  g_hash_table_insert (_qs_table, g_strdup (id), data);
  G_UNLOCK (query_server_table);

  return data;
}


/**
 * @brief Get edge handle from server data.
 */
nns_edge_h
gst_tensor_query_server_get_edge_handle (edge_server_handle server_h)
{
  GstTensorQueryServer *data = (GstTensorQueryServer *) server_h;

  return data->edge_h;
}

/**
 * @brief Remove GstTensorQueryServer.
 */
void
gst_tensor_query_server_remove_data (edge_server_handle server_h)
{
  GstTensorQueryServer *data = (GstTensorQueryServer *) server_h;

  if (NULL == data) {
    return;
  }

  G_LOCK (query_server_table);
  if (!g_hash_table_lookup (_qs_table, data->id))
    g_hash_table_remove (_qs_table, data->id);
  G_UNLOCK (query_server_table);

  if (data->edge_h) {
    nns_edge_release_handle (data->edge_h);
    data->edge_h = NULL;
  }

  g_cond_clear (&data->cond);
  g_mutex_clear (&data->lock);
  g_free (data);
  data = NULL;
}

/**
 * @brief Wait until the sink is configured and get server info handle.
 */
gboolean
gst_tensor_query_server_wait_sink (edge_server_handle server_h)
{
  gint64 end_time;
  GstTensorQueryServer *data = (GstTensorQueryServer *) server_h;

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
gst_tensor_query_server_set_configured (edge_server_handle server_h)
{
  GstTensorQueryServer *data = (GstTensorQueryServer *) server_h;
  if (NULL == data) {
    return;
  }
  g_mutex_lock (&data->lock);
  data->configured = TRUE;
  g_cond_broadcast (&data->cond);
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
  _qs_table = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, NULL);
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
