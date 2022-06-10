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

#include <unistd.h>
#include <MQTTAsync.h>
#include "nnstreamer_edge_common.h"
#include "nnstreamer_edge_internal.h"

/**
 * @brief Callback function to be called when the connection is lost.
 */
static void
mqtt_cb_connection_lost (void *context, char *cause)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  nns_edge_logw ("MQTT connection is lost (ID:%s, Cause:%s).", eh->id, cause);
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

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  nns_edge_logi ("MQTT connection is completed (ID:%s).", eh->id);
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

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  nns_edge_logw ("MQTT connection is failed (ID:%s).", eh->id);
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

  UNUSED (response);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return;
  }

  nns_edge_logi ("MQTT disconnection is completed (ID:%s).", eh->id);
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

  UNUSED (topic);
  UNUSED (topic_len);
  UNUSED (message);
  eh = (nns_edge_handle_s *) context;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return TRUE;
  }

  nns_edge_logd ("MQTT message is arrived (ID:%s, Topic:%s).",
      eh->id, eh->topic);
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
  MQTTAsync handle;
  MQTTAsync_connectOptions options = MQTTAsync_connectOptions_initializer;
  char *url;
  char *client_id;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_logi ("Trying to connect MQTT (ID:%s, URL:%s:%d).",
      eh->id, eh->ip, eh->port);

  url = g_strdup_printf ("%s:%d", eh->ip, eh->port);
  client_id = g_strdup_printf ("nns_edge_%s_%u", eh->id, getpid ());

  ret = MQTTAsync_create (&handle, url, client_id,
      MQTTCLIENT_PERSISTENCE_NONE, NULL);
  if (MQTTASYNC_SUCCESS != ret) {
    nns_edge_loge ("Failed to create MQTT handle.");
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }

  MQTTAsync_setCallbacks (handle, edge_h,
      mqtt_cb_connection_lost, mqtt_cb_message_arrived, NULL);

  options.cleansession = 1;
  options.keepAliveInterval = 6;
  options.onSuccess = mqtt_cb_connection_success;
  options.onFailure = mqtt_cb_connection_failure;
  options.context = edge_h;

  if (MQTTAsync_connect (handle, &options) != MQTTASYNC_SUCCESS) {
    nns_edge_loge ("Failed to connect MQTT.");
    MQTTAsync_destroy (&handle);
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }

  eh->mqtt_handle = handle;
  ret = NNS_EDGE_ERROR_NONE;

error:
  g_free (url);
  g_free (client_id);
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
  MQTTAsync handle;
  MQTTAsync_disconnectOptions options = MQTTAsync_disconnectOptions_initializer;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  handle = eh->mqtt_handle;

  if (!handle) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
    return NNS_EDGE_ERROR_IO;
  }

  nns_edge_logi ("Trying to disconnect MQTT (ID:%s, URL:%s:%d).",
      eh->id, eh->ip, eh->port);

  options.onSuccess = mqtt_cb_disconnection_success;
  options.onFailure = mqtt_cb_disconnection_failure;
  options.context = edge_h;

  while (MQTTAsync_isConnected (handle)) {
    if (MQTTAsync_disconnect (handle, &options) != MQTTASYNC_SUCCESS) {
      nns_edge_loge ("Failed to disconnect MQTT.");
      return NNS_EDGE_ERROR_IO;
    }
  }

  MQTTAsync_destroy (&handle);
  eh->mqtt_handle = NULL;

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
  MQTTAsync handle;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data || length <= 0) {
    nns_edge_loge ("Invalid param, given data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  handle = eh->mqtt_handle;

  if (!handle || MQTTAsync_isConnected (handle)) {
    nns_edge_loge ("Invalid state, MQTT connection was not completed.");
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
  MQTTAsync handle;
  int ret;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  handle = eh->mqtt_handle;

  if (!handle || MQTTAsync_isConnected (handle)) {
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
