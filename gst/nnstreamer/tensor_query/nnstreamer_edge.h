/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   nnstreamer_edge.h
 * @date   25 Mar 2022
 * @brief  Common library to support communication among devices.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note This file will be moved to nnstreamer-edge repo. (https://github.com/nnstreamer/nnstreamer-edge)
 */

#ifndef __NNSTREAER_EDGE_H__
#define __NNSTREAER_EDGE_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stddef.h>
#include <stdbool.h>

typedef void *nns_edge_h;
typedef void *nns_edge_data_h;

typedef enum
{
  NNS_EDGE_PROTOCOL_TCP = 0,
  NNS_EDGE_PROTOCOL_MQTT,
  NNS_EDGE_PROTOCOL_AITT,
  NNS_EDGE_PROTOCOL_AITT_TCP,

  NNS_EDGE_PROTOCOL_MAX
} nns_edge_protocol_e;

typedef enum
{
  NNS_EDGE_DATA_TYPE_INFO = 0,
  NNS_EDGE_DATA_TYPE_RAW_DATA,

  NNS_EDGE_DATA_TYPE_MAX
} nns_edge_data_type_e;

/**
 * @brief Get registered handle. If not registered, create new handle and register it.
 */
nns_edge_h
nns_edge_get_handle (const char *id, nns_edge_protocol_e protocol);

/**
 * @brief Release the given handle.
 */
void
nns_edge_release_handle (nns_edge_h edge_h);

/**
 * @brief Connect to the broker.
 */
int
nns_edge_connect (nns_edge_h edge_h, const char *ip, int port);

/**
 * @brief Disconnect from the broker.
 */
int
nns_edge_disconnect (nns_edge_h edge_h);

/**
 * @brief Create nnstreamer edge data.
 */
void
nns_edge_create_data (nns_edge_data_h *data_h);

/**
 * @brief Set nnstreamer edge data.
 */
void
nns_edge_set_data (nns_edge_data_h data_h, const char *topic, const void *data, const size_t data_len, nns_edge_data_type_e dtype);

/**
 * @brief Destroy nnstreamer edge data.
 */
void
nns_edge_destroy_data (nns_edge_data_h data_h);

/**
 * @brief Publish a message to a given topic.
 */
int
nns_edge_publish (nns_edge_h edge_h, nns_edge_data_h data_h);

/**
 * @brief Request result to the server.
 */
int
nns_edge_request (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data);

/**
 * @brief Get topic from the message.
 */
const char *
nns_edge_get_topic (nns_edge_h edge_h);

/**
 * @brief Subscribe a message to a given topic.
 */
int
nns_edge_subscribe (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data);

/**
 * @brief Unsubscribe a message to a given topic.
 */
int
nns_edge_unsubscribe (nns_edge_h edge_h);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __NNSTREAER_EDGE_H__ */
