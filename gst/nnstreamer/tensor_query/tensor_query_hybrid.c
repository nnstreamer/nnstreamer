/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_hybrid.c
 * @date   12 Oct 2021
 * @brief  Utility functions for tensor-query hybrid feature
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include "tensor_query_hybrid.h"
#include "nnstreamer_log.h"

#if defined(ENABLE_QUERY_HYBRID)
#include <nnsquery.h>

/**
 * @brief Internal function to free node info.
 */
static void
_free_node_info (query_node_info_s * node)
{
  if (node) {
    g_free (node->host);
    node->host = NULL;
  }
}

/**
 * @brief Internal function to free server info.
 */
static void
_free_server_info (query_server_info_s * server)
{
  if (server) {
    _free_node_info (&server->src);
    _free_node_info (&server->sink);
  }
}

/**
 * @brief Parse received message.
 */
static void
_broker_parse_message (query_hybrid_info_s * info, gchar * payload)
{
  gchar **splits;
  query_server_info_s *server;

  server = g_try_new0 (query_server_info_s, 1);
  if (!server) {
    nns_loge ("[Query-hybrid] Failed to allocate query server info.");
    return;
  }

  splits = g_strsplit (payload, "/", -1);
  server->src.host = g_strdup (splits[0]);
  server->src.port = g_ascii_strtoull (splits[1], NULL, 10);
  server->sink.host = g_strdup (splits[2]);
  server->sink.port = g_ascii_strtoull (splits[3], NULL, 10);

  nns_logd ("[Query-hybrid] Parsed info: src[%s:%u] sink[%s:%u]",
      server->src.host, server->src.port, server->sink.host, server->sink.port);

  g_async_queue_push (info->server_list, server);
  g_strfreev (splits);
}

/**
 * @brief State change callback.
 */
static void
_broker_state_change_cb (void *user_data, query_mqtt_state_t state)
{
  query_hybrid_info_s *info = (query_hybrid_info_s *) user_data;

  info->state = (gint) state;
  nns_logd ("[Query-hybrid] Broker state changed to %d.", state);
}

/**
 * @brief Raw message received callback.
 */
static void
_broker_msg_received_cb (const gchar * topic, msg_data * msg, gint msg_len,
    void *user_data)
{
  query_hybrid_info_s *info = (query_hybrid_info_s *) user_data;
  gchar *payload;
  gint size;

  if (msg_len <= 0) {
    nns_logd ("[Query-hybrid] There is no data to receive from broker.");
    return;
  }

  size = msg_len - sizeof (msg->type);
  payload = (gchar *) g_malloc0 (size + 1);
  memcpy (payload, msg->payload, size);
  payload[size] = '\0';

  nns_logd ("[Query-hybrid] Received topic: %s (size: %d)\n", topic, msg_len);
  nns_logd ("[Query-hybrid] payload: %s", payload);

  _broker_parse_message (info, payload);
  g_free (payload);
}

/**
 * @brief Connect to broker.
 */
static gboolean
_broker_connect (query_hybrid_info_s * info)
{
  gchar *host, *port;
  gint err;

  host = g_strdup (info->broker.host);
  port = g_strdup_printf ("%u", info->broker.port);

  err = query_open_connection (&info->handle, host, port,
      _broker_state_change_cb, info);
  if (err == 0) {
    /* Wait until connection is established. */
    while (MQTT_CONNECTED != info->state) {
      g_usleep (10000);
    }
  }

  g_free (host);
  g_free (port);

  if (err != 0) {
    nns_loge ("[Query-hybrid] Failed to connect broker (error: %d).", err);
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Initialize query-hybrid info.
 */
void
tensor_query_hybrid_init (query_hybrid_info_s * info,
    const gchar * host, const guint16 port, gboolean is_server)
{
  g_return_if_fail (info != NULL);

  memset (info, 0, sizeof (query_hybrid_info_s));
  info->node.host = g_strdup (host);
  info->node.port = port;
  info->is_server = is_server;
  info->state = MQTT_DISCONNECTED;

  if (!is_server) {
    info->server_list = g_async_queue_new ();
  }
}

/**
 * @brief Close connection and free query-hybrid info.
 */
void
tensor_query_hybrid_close (query_hybrid_info_s * info)
{
  g_return_if_fail (info != NULL);

  if (info->handle) {
    if (info->topic) {
      query_clear_retained_topic (info->handle, info->topic);
      g_free (info->topic);
      info->topic = NULL;
    }

    if (0 != query_close_connection (info->handle)) {
      nns_loge ("[Query-hybrid] Failed to close broker connection.");
    }

    info->handle = NULL;
  }

  if (info->server_list) {
    gpointer data;

    while ((data = g_async_queue_try_pop (info->server_list))) {
      tensor_query_hybrid_free_server_info (data);
    }

    g_async_queue_unref (info->server_list);
    info->server_list = NULL;
  }

  _free_node_info (&info->broker);
  _free_node_info (&info->node);
}

/**
 * @brief Set current node info.
 */
void
tensor_query_hybrid_set_node (query_hybrid_info_s * info, const gchar * host,
    const guint16 port, query_server_info_handle server_info_h)
{
  g_return_if_fail (info != NULL);

  _free_node_info (&info->node);
  info->node.host = g_strdup (host);
  info->node.port = port;
  info->node.server_info_h = server_info_h;
}

/**
 * @brief Set broker info.
 */
void
tensor_query_hybrid_set_broker (query_hybrid_info_s * info,
    const gchar * host, const guint16 port)
{
  g_return_if_fail (info != NULL);

  _free_node_info (&info->broker);
  info->broker.host = g_strdup (host);
  info->broker.port = port;
}

/**
 * @brief Get server info from node list.
 * @return Server node info. Caller should release server info using tensor_query_hybrid_free_server_info().
 */
query_server_info_s *
tensor_query_hybrid_get_server_info (query_hybrid_info_s * info)
{
  query_server_info_s *server = NULL;

  g_return_val_if_fail (info != NULL, FALSE);

  if (info->is_server) {
    nns_logw ("[Query-hybrid] It is server node, cannot get the server info.");
  } else {
    /**
     * @todo Need to update server selection policy. Now, use first received info.
     */
    server = (query_server_info_s *) g_async_queue_try_pop (info->server_list);
  }

  return server;
}

/**
 * @brief Free server info.
 */
void
tensor_query_hybrid_free_server_info (query_server_info_s * server)
{
  if (server) {
    _free_server_info (server);
    g_free (server);
  }
}

/**
 * @brief Connect to broker and publish topic.
 */
gboolean
tensor_query_hybrid_publish (query_hybrid_info_s * info,
    const gchar * operation)
{
  gchar *device, *topic, *msg;
  gchar *sink_host = NULL;
  guint16 sink_port;
  gint err;

  g_return_val_if_fail (info != NULL, FALSE);
  g_return_val_if_fail (operation != NULL, FALSE);

  if (!info->is_server) {
    nns_logw ("[Query-hybrid] It is client node, cannot publish topic.");
    return FALSE;
  }

  if (!info->node.host || info->node.port == 0) {
    nns_loge ("[Query-hybrid] Invalid node info, cannot publish topic.");
    return FALSE;
  }

  if (!_broker_connect (info)) {
    nns_loge ("[Query-hybrid] Failed to publish, connection failed.");
    return FALSE;
  }

  /**
   * @todo Set unique device name.
   * Device name should be unique. Consider using MAC address later.
   * Now, use IP and port number temporarily.
   */
  device = g_strdup_printf ("device-%s-%u", info->node.host, info->node.port);
  topic = g_strdup_printf ("edge/inference/%s/%s/", device, operation);
  nns_logd ("[Query-hybrid] Query server topic: %s", topic);
  g_free (device);

  sink_host = gst_tensor_query_server_get_sink_host (info->node.server_info_h);
  if (!sink_host) {
    nns_logw
        ("[Query-hybrid] Sink host is not given. Use default host (localhost).");
    sink_host = g_strdup ("localhost");
  }
  sink_port = gst_tensor_query_server_get_sink_port (info->node.server_info_h);
  if (0 == sink_port) {
    nns_logw
        ("[Query-hybrid] Sink port is not given. Use default port (3000).");
    sink_port = 3000;
  }

  msg = g_strdup_printf ("%s/%u/%s/%u", info->node.host, info->node.port,
      sink_host, sink_port);
  g_free (sink_host);
  nns_logd ("[Query-hybrid] Query server source msg: %s", msg);

  err = query_publish_raw_data (info->handle, topic, msg, strlen (msg), TRUE);
  g_free (msg);

  if (err != 0) {
    nns_loge ("[Query-hybrid] Failed to publish raw data (error: %d).", err);
    g_free (topic);
    return FALSE;
  }

  info->topic = topic;
  return TRUE;
}

/**
 * @brief Connect to broker and subcribe topic.
 */
gboolean
tensor_query_hybrid_subscribe (query_hybrid_info_s * info,
    const gchar * operation)
{
  gchar *topic;
  gint err;

  g_return_val_if_fail (info != NULL, FALSE);
  g_return_val_if_fail (operation != NULL, FALSE);

  if (info->is_server) {
    nns_logw ("[Query-hybrid] It is server node, cannot subcribe topic.");
    return FALSE;
  }

  if (!_broker_connect (info)) {
    nns_loge ("[Query-hybrid] Failed to subscribe, connection failed.");
    return FALSE;
  }

  topic = g_strdup_printf ("edge/inference/+/%s/#", operation);
  err = query_subscribe_topic (info->handle, topic,
      _broker_msg_received_cb, info);
  g_free (topic);

  if (err != 0) {
    nns_loge ("[Query-hybrid] Failed to subscribe topic (error: %d).", err);
    return FALSE;
  }

  return TRUE;
}
#endif /* ENABLE_QUERY_HYBRID */
