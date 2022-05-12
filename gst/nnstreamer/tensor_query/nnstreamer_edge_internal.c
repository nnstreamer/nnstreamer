/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_internal.c
 * @date   6 April 2022
 * @brief  Common library to support communication among devices.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "nnstreamer_edge_common.h"

/**
 * @brief Data structure for edge handle.
 */
typedef struct
{
  unsigned int magic;
  char *id;
  char *topic;
  nns_edge_protocol_e protocol;
  char *ip;
  int port;
  nns_edge_event_cb event_cb;
  void *user_data;
} nns_edge_handle_s;

/**
 * @brief Validate edge handle.
 */
bool
nns_edge_handle_is_valid (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (eh))
    return false;

  return true;
}

/**
 * @brief Check network connection.
 */
bool
nns_edge_is_connected (nns_edge_h edge_h)
{
  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return false;
  }

  /** @todo check connection status */
  return true;
}

/**
 * @brief Get registered handle. If not registered, create new handle and register it.
 */
int
nns_edge_get_handle (const char *id, const char *topic, nns_edge_h * edge_h)
{
  nns_edge_handle_s *eh;

  if (!id || *id == '\0') {
    nns_edge_loge ("Invalid param, given ID is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!topic || *topic == '\0') {
    nns_edge_loge ("Invalid param, given topic is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!edge_h) {
    nns_edge_loge ("Invalid param, edge_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /**
   * @todo manage edge handles
   * 1. consider adding hash table or list to manage edge handles.
   * 2. compare topic and return error if existing topic in handle is different.
   */
  eh = (nns_edge_handle_s *) malloc (sizeof (nns_edge_handle_s));
  if (!eh) {
    nns_edge_loge ("Failed to allocate memory for edge handle.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (eh, 0, sizeof (nns_edge_handle_s));
  eh->magic = NNS_EDGE_MAGIC;
  eh->id = g_strdup (id);
  eh->topic = g_strdup (topic);
  eh->protocol = NNS_EDGE_PROTOCOL_MAX;

  *edge_h = eh;
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Release the given handle.
 */
int
nns_edge_release_handle (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;

  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (eh->event_cb) {
    /** @todo send new event (release handle) */
    eh->event_cb = NULL;
  }

  eh->magic = NNS_EDGE_MAGIC_DEAD;
  g_free (eh->id);
  g_free (eh->topic);
  g_free (eh->ip);
  g_free (eh);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set the event callback.
 */
int
nns_edge_set_event_callback (nns_edge_h edge_h, nns_edge_event_cb cb,
    void *user_data)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;

  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (eh->event_cb) {
    /** @todo send new event (release callback) */
  }

  eh->event_cb = cb;
  eh->user_data = user_data;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Connect to the destination node.
 */
int
nns_edge_connect (nns_edge_h edge_h, nns_edge_protocol_e protocol,
    const char *ip, int port)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;

  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!ip || *ip == '\0') {
    nns_edge_loge ("Invalid param, given IP is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (nns_edge_is_connected (edge_h)) {
    nns_edge_loge ("Already connected to %s:%d.", eh->ip, eh->port);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  eh->protocol = protocol;
  eh->ip = g_strdup (ip);
  eh->port = port;

  /** @todo update code for connection */
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Disconnect from the destination node.
 */
int
nns_edge_disconnect (nns_edge_h edge_h)
{
  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (nns_edge_is_connected (edge_h)) {
    /** @todo update code for disconnection */
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Publish a message to a given topic.
 */
int
nns_edge_publish (nns_edge_h edge_h, nns_edge_data_h data_h)
{
  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!nns_edge_data_is_valid (data_h)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!nns_edge_is_connected (edge_h)) {
    nns_edge_loge ("Connection failure.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  /** @todo update code (publish data) */
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Request result to the server.
 */
int
nns_edge_request (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data)
{
  UNUSED (user_data);

  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!nns_edge_data_is_valid (data_h)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!nns_edge_is_connected (edge_h)) {
    nns_edge_loge ("Connection failure.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  /** @todo update code (request - send, wait for response) */
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Subscribe a message to a given topic.
 */
int
nns_edge_subscribe (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data)
{
  UNUSED (data_h);
  UNUSED (user_data);

  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!nns_edge_is_connected (edge_h)) {
    nns_edge_loge ("Connection failure.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  /** @todo update code (subscribe) */
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Unsubscribe a message to a given topic.
 */
int
nns_edge_unsubscribe (nns_edge_h edge_h)
{
  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!nns_edge_is_connected (edge_h)) {
    nns_edge_loge ("Connection failure.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  /** @todo update code (unsubscribe) */
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the topic of edge handle. Caller should release returned string using free().
 * @todo is this necessary?
 */
int
nns_edge_get_topic (nns_edge_h edge_h, char **topic)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;

  if (!nns_edge_handle_is_valid (edge_h)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!topic) {
    nns_edge_loge ("Invalid param, topic should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *topic = g_strdup (eh->topic);

  return NNS_EDGE_ERROR_NONE;
}
