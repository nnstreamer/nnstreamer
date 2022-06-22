/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_internal.h
 * @date   11 May 2022
 * @brief  Internal functions to support communication among devices.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file is internal header for nnstreamer edge. DO NOT export this file.
 */

#ifndef __NNSTREAMER_EDGE_INTERNAL_H__
#define __NNSTREAMER_EDGE_INTERNAL_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "nnstreamer_edge.h"
#include <gio/gio.h>
#include <netinet/tcp.h>
#include <netinet/in.h>

/**
 * @brief Data structure for edge handle.
 */
typedef struct {
  unsigned int magic;
  pthread_mutex_t lock;
  char *id;
  char *topic;
  nns_edge_protocol_e protocol;
  char *ip;
  int port;

  /* Edge event callback and user data */
  nns_edge_event_cb event_cb;
  void *user_data;

  bool is_server;
  int64_t client_id;
  char *caps_str;
  char *recv_ip;
  int recv_port;
  GHashTable *conn_table;

  GSocketListener *listener;
  GCancellable *cancellable;

  /* MQTT */
  void *mqtt_handle;
} nns_edge_handle_s;

#if defined(ENABLE_MQTT)
/**
 * @brief Connect to MQTT.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int nns_edge_mqtt_connect (nns_edge_h edge_h);

/**
 * @brief Close the connection to MQTT.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int nns_edge_mqtt_close (nns_edge_h edge_h);

/**
 * @brief Publish raw data.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int nns_edge_mqtt_publish (nns_edge_h edge_h, const void *data, const int length);

/**
 * @brief Subscribe a topic.
 * @note This is internal function for MQTT broker. You should call this with edge-handle lock.
 */
int nns_edge_mqtt_subscribe (nns_edge_h edge_h);
#else
/**
 * @todo consider to change code style later.
 * If MQTT is disabled, nnstreamer does not include nnstreamer_edge_mqtt.c, and changing code style will make error as it is not used function now.
 *
 * static int nns_edge_mqtt_publish (nns_edge_h edge_h, const void *data, const int length)
 * {
 *   return NNS_EDGE_ERROR_NOT_SUPPORTED;
 * }
 */
#define nns_edge_mqtt_connect(...) (NNS_EDGE_ERROR_NOT_SUPPORTED)
#define nns_edge_mqtt_close(...) (NNS_EDGE_ERROR_NOT_SUPPORTED)
#define nns_edge_mqtt_publish(...) (NNS_EDGE_ERROR_NOT_SUPPORTED)
#define nns_edge_mqtt_subscribe(...) (NNS_EDGE_ERROR_NOT_SUPPORTED)
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __NNSTREAMER_EDGE_INTERNAL_H__ */
