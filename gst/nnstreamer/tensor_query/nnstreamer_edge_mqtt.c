/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_mqtt.c
 * @date   11 May 2022
 * @brief  Internal functions to support MQTT protocol (Paho Asynchronous MQTT C Client Library).
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Sangjung Woo <sangjung.woo@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#if !defined(ENABLE_MQTT)
#error "This file can be built with Paho MQTT library."
#endif
#define DEFAULT_SUB_TIMEOUT 1000000 /** 1 second */

#include <unistd.h>
#include <MQTTAsync.h>
#include "nnstreamer-edge-common.h"
#include "nnstreamer-edge-internal.h"

/**
 * @brief Data structure for mqtt broker handle.
 */
typedef struct
{
  void *mqtt_h;
  GAsyncQueue *server_list;
  GMutex mqtt_mutex;
  GCond mqtt_gcond;
  gboolean mqtt_is_connected;
} nns_edge_broker_s;

/**
 * @brief Callback function to be called when the connection is lost.
 */
static void
mqtt_cb_connection_lost (void *context, char *cause)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;

  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;
  nns_edge_logw ("MQTT connection is lost (ID:%s, Cause:%s).", eh->id, cause);
  g_mutex_lock (&bh->mqtt_mutex);
  bh->mqtt_is_connected = FALSE;
  g_cond_broadcast (&bh->mqtt_gcond);
  g_mutex_unlock (&bh->mqtt_mutex);

  if (eh->event_cb) {
    /** @todo send new event (MQTT disconnected) */
  }
}

/**
 * @brief Callback function to be called when the connection is completed.
 */
static void
mqtt_cb_connection_success (void *context, MQTTAsync_successData * response)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;

  g_mutex_lock (&bh->mqtt_mutex);
  bh->mqtt_is_connected = TRUE;
  g_cond_broadcast (&bh->mqtt_gcond);
  g_mutex_unlock (&bh->mqtt_mutex);

  if (eh->event_cb) {
    /** @todo send new event (MQTT connected) */
  }
}

/**
 * @brief Callback function to be called when the connection is failed.
 */
static void
mqtt_cb_connection_failure (void *context, MQTTAsync_failureData * response)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;

  nns_edge_logw ("MQTT connection is failed (ID:%s).", eh->id);
  g_mutex_lock (&bh->mqtt_mutex);
  bh->mqtt_is_connected = FALSE;
  g_cond_broadcast (&bh->mqtt_gcond);
  g_mutex_unlock (&bh->mqtt_mutex);

  if (eh->event_cb) {
    /** @todo send new event (MQTT connection failure) */
  }
}

/**
 * @brief Callback function to be called when the disconnection is completed.
 */
static void
mqtt_cb_disconnection_success (void *context, MQTTAsync_successData * response)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;

  nns_edge_logi ("MQTT disconnection is completed (ID:%s).", eh->id);
  g_mutex_lock (&bh->mqtt_mutex);
  bh->mqtt_is_connected = FALSE;
  g_cond_broadcast (&bh->mqtt_gcond);
  g_mutex_unlock (&bh->mqtt_mutex);

  if (eh->event_cb) {
    /** @todo send new event (MQTT disconnected) */
  }
}

/**
 * @brief Callback function to be called when the disconnection is failed.
 */
static void
mqtt_cb_disconnection_failure (void *context, MQTTAsync_failureData * response)
{
  nns_edge_handle_s *eh;

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  nns_edge_logw ("MQTT disconnection is failed (ID:%s).", eh->id);
  if (eh->event_cb) {
    /** @todo send new event (MQTT disconnection failure) */
  }
}

/**
 * @brief Callback function to be called when a message is arrived.
 * @return Return TRUE to prevent delivering the message again.
 */
static int
mqtt_cb_message_arrived (void *context, char *topic, int topic_len,
    MQTTAsync_message * message)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;
  char *msg = NULL;

  UNUSED (topic);
  UNUSED (topic_len);
  UNUSED (message);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return TRUE;
  }

  if (0 >= message->payloadlen) {
    nns_edge_logw ("Invalid payload lenth: %d", message->payloadlen);
    return TRUE;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;

  nns_edge_logd ("MQTT message is arrived (ID:%s, Topic:%s).",
      eh->id, eh->topic);

  msg = (char *) malloc (message->payloadlen);
  memcpy (msg, message->payload, message->payloadlen);
  g_async_queue_push (bh->server_list, msg);

  if (eh->event_cb) {
    /** @todo send new event (message arrived) */
  }

  return TRUE;
}

/**
 * @brief Connect to MQTT.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int
nns_edge_mqtt_connect (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;
  MQTTAsync_connectOptions options = MQTTAsync_connectOptions_initializer;
  int ret = NNS_EDGE_ERROR_NONE;
  int64_t end_time;
  MQTTAsync handle;
  char *url;
  char *client_id;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  bh = (nns_edge_broker_s *) malloc (sizeof (nns_edge_broker_s));
  if (!bh) {
    nns_edge_loge ("Failed to allocate memory for broker handle.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (bh, 0, sizeof (nns_edge_broker_s));

  url = g_strdup_printf ("%s:%d", eh->ip, eh->port);
  client_id = g_strdup_printf ("nns_edge_%s_%u", eh->id, getpid ());

  ret = MQTTAsync_create (&handle, url, client_id,
      MQTTCLIENT_PERSISTENCE_NONE, NULL);
  if (MQTTASYNC_SUCCESS != ret) {
    nns_edge_loge ("Failed to create MQTT handle.");
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }

  g_cond_init (&bh->mqtt_gcond);
  g_mutex_init (&bh->mqtt_mutex);
  bh->mqtt_is_connected = FALSE;
  bh->mqtt_h = handle;
  bh->server_list = g_async_queue_new ();
  eh->broker_h = bh;

  bh = (nns_edge_broker_s *) eh->broker_h;
  if (!bh->mqtt_h) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    ret = NNS_EDGE_ERROR_IO;
    goto error;
  }
  handle = bh->mqtt_h;

  nns_edge_logi ("Trying to connect MQTT (ID:%s, URL:%s:%d).",
      eh->id, eh->ip, eh->port);

  MQTTAsync_setCallbacks (handle, edge_h,
      mqtt_cb_connection_lost, mqtt_cb_message_arrived, NULL);

  options.cleansession = 1;
  options.keepAliveInterval = 6;
  options.onSuccess = mqtt_cb_connection_success;
  options.onFailure = mqtt_cb_connection_failure;
  options.context = edge_h;

  if (MQTTAsync_connect (handle, &options) != MQTTASYNC_SUCCESS) {
    nns_edge_loge ("Failed to connect MQTT.");
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }

  /* Waiting for the connection */
  end_time = g_get_monotonic_time () + 5 * G_TIME_SPAN_SECOND;
  g_mutex_lock (&bh->mqtt_mutex);
  while (!bh->mqtt_is_connected) {
    if (!g_cond_wait_until (&bh->mqtt_gcond, &bh->mqtt_mutex, end_time)) {
      g_mutex_unlock (&bh->mqtt_mutex);
      nns_edge_loge ("Failed to connect to MQTT broker."
          "Please check broker is running status or broker host address.");
      goto error;
    }
  }
  g_mutex_unlock (&bh->mqtt_mutex);
  return NNS_EDGE_ERROR_NONE;

error:
  nns_edge_mqtt_close (eh);
  return ret;
}

/**
 * @brief Close the connection to MQTT.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int
nns_edge_mqtt_close (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;
  MQTTAsync handle;
  MQTTAsync_disconnectOptions options = MQTTAsync_disconnectOptions_initializer;
  char *msg;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;

  if (!bh->mqtt_h) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  handle = bh->mqtt_h;

  nns_edge_logi ("Trying to disconnect MQTT (ID:%s, URL:%s:%d).",
      eh->id, eh->ip, eh->port);

  options.onSuccess = mqtt_cb_disconnection_success;
  options.onFailure = mqtt_cb_disconnection_failure;
  options.context = edge_h;

  /** Clear retained message */
  MQTTAsync_send (handle, eh->topic, 0, NULL, 1, 1, NULL);

  while (MQTTAsync_isConnected (handle)) {
    if (MQTTAsync_disconnect (handle, &options) != MQTTASYNC_SUCCESS) {
      nns_edge_loge ("Failed to disconnect MQTT.");
      return NNS_EDGE_ERROR_IO;
    }
    g_usleep (10000);
  }
  g_cond_clear (&bh->mqtt_gcond);
  g_mutex_clear (&bh->mqtt_mutex);

  MQTTAsync_destroy (&handle);

  while ((msg = g_async_queue_try_pop (bh->server_list))) {
    SAFE_FREE (msg);
  }
  g_async_queue_unref (bh->server_list);
  bh->server_list = NULL;
  SAFE_FREE (bh);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Publish raw data.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int
nns_edge_mqtt_publish (nns_edge_h edge_h, const void *data, const int length)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;
  MQTTAsync handle;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data || length <= 0) {
    nns_edge_loge ("Invalid param, given data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;
  if (!bh->mqtt_h) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  handle = bh->mqtt_h;

  if (!MQTTAsync_isConnected (handle)) {
    nns_edge_loge ("Failed to publish message, MQTT is not connected.");
    return NNS_EDGE_ERROR_IO;
  }

  /* Publish a message (default QoS 1 - at least once and retained true). */
  ret = MQTTAsync_send (handle, eh->topic, length, data, 1, 1, NULL);
  if (ret != MQTTASYNC_SUCCESS) {
    nns_edge_loge ("Failed to publish a message (ID:%s, Topic:%s).",
        eh->id, eh->topic);
    return NNS_EDGE_ERROR_IO;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Subscribe a topic.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int
nns_edge_mqtt_subscribe (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;
  MQTTAsync handle;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;
  if (!bh->mqtt_h) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  handle = bh->mqtt_h;

  if (!MQTTAsync_isConnected (handle)) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    return NNS_EDGE_ERROR_IO;
  }

  /* Subscribe a topic (default QoS 1 - at least once). */
  ret = MQTTAsync_subscribe (handle, eh->topic, 1, NULL);
  if (ret != MQTTASYNC_SUCCESS) {
    nns_edge_loge ("Failed to subscribe a topic (ID:%s, Topic:%s).",
        eh->id, eh->topic);
    return NNS_EDGE_ERROR_IO;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Check mqtt connection
 */
bool
nns_edge_mqtt_is_connected (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;
  MQTTAsync handle;
  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return false;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;
  if (!bh->mqtt_h) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    return false;
  }
  handle = bh->mqtt_h;

  if (MQTTAsync_isConnected (handle)) {
    return true;
  }

  return false;
}

/**
 * @brief Get message from mqtt broker.
 */
int
nns_edge_mqtt_get_message (nns_edge_h edge_h, char **msg)
{
  nns_edge_handle_s *eh;
  nns_edge_broker_s *bh;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh) || !eh->broker_h) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  bh = (nns_edge_broker_s *) eh->broker_h;

  *msg = g_async_queue_timeout_pop (bh->server_list, DEFAULT_SUB_TIMEOUT);
  if (!*msg) {
    nns_edge_loge ("Failed to get message from mqtt broker within timeout");
    return NNS_EDGE_ERROR_UNKNOWN;
  }

  return NNS_EDGE_ERROR_NONE;
}
