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
 *
 * @todo Update document and sample code.
 * 1. Add sample code when the 1st API set is complete - connection, pub/sub, request, ...
 * 2. Update license when migrating this into edge repo. (Apache-2.0)
 */

#ifndef __NNSTREAMER_EDGE_H__
#define __NNSTREAMER_EDGE_H__

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void *nns_edge_h;
typedef void *nns_edge_event_h;
typedef void *nns_edge_data_h;

/**
 * @brief The maximum number of data instances that nnstreamer-edge data may have.
 */
#define NNS_EDGE_DATA_LIMIT (16)

/**
 * @brief Enumeration for the error codes of nnstreamer-edge.
 * @todo define detailed error code later (linux standard error, sync with tizen error code)
 */
typedef enum {
  NNS_EDGE_ERROR_NONE = 0,
  NNS_EDGE_ERROR_INVALID_PARAMETER = -EINVAL,
  NNS_EDGE_ERROR_OUT_OF_MEMORY = -ENOMEM,
  NNS_EDGE_ERROR_IO = -EIO,
  NNS_EDGE_ERROR_CONNECTION_FAILURE = -ECONNREFUSED,
  NNS_EDGE_ERROR_UNKNOWN = (-1073741824LL),
  NNS_EDGE_ERROR_NOT_SUPPORTED = (NNS_EDGE_ERROR_UNKNOWN + 2),
} nns_edge_error_e;

typedef enum {
  NNS_EDGE_EVENT_UNKNOWN = 0,
  NNS_EDGE_EVENT_CAPABILITY,
  NNS_EDGE_EVENT_NEW_DATA_RECEIVED,
  NNS_EDGE_EVENT_CALLBACK_RELEASED,

  NNS_EDGE_EVENT_CUSTOM = 0x01000000
} nns_edge_event_e;

typedef enum {
  NNS_EDGE_PROTOCOL_TCP = 0,
  NNS_EDGE_PROTOCOL_MQTT,
  NNS_EDGE_PROTOCOL_AITT,
  NNS_EDGE_PROTOCOL_AITT_TCP,

  NNS_EDGE_PROTOCOL_MAX
} nns_edge_protocol_e;

/**
 * @brief Callback for the nnstreamer edge event.
 * @note This callback will suspend data stream. Do not spend too much time in the callback.
 * @return User should return NNS_EDGE_ERROR_NONE if an event is successfully handled.
 */
typedef int (*nns_edge_event_cb) (nns_edge_event_h event_h, void *user_data);

/**
 * @brief Callback called when nnstreamer-edge data is released.
 */
typedef void (*nns_edge_data_destroy_cb) (void *data);

/**
 * @brief Get registered handle. If not registered, create new handle and register it.
 */
int nns_edge_create_handle (const char *id, const char *topic, nns_edge_h *edge_h);

/**
 * @brief Start the nnstreamer edge.
 */
int nns_edge_start (nns_edge_h edge_h, bool is_server);

/**
 * @brief Release the given handle.
 */
int nns_edge_release_handle (nns_edge_h edge_h);

/**
 * @brief Set the event callback.
 */
int nns_edge_set_event_callback (nns_edge_h edge_h, nns_edge_event_cb cb, void *user_data);

/**
 * @brief Connect to the destination node.
 */
int nns_edge_connect (nns_edge_h edge_h, nns_edge_protocol_e protocol, const char *ip, int port);

/**
 * @brief Disconnect from the destination node.
 */
int nns_edge_disconnect (nns_edge_h edge_h);

/**
 * @brief Publish a message to a given topic.
 */
int nns_edge_publish (nns_edge_h edge_h, nns_edge_data_h data_h);

/**
 * @brief Request result to the server.
 */
int nns_edge_request (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data);

/**
 * @brief Respond to a request.
 */
int nns_edge_respond (nns_edge_h edge_h, nns_edge_data_h data_h);

/**
 * @brief Subscribe a message to a given topic.
 */
int nns_edge_subscribe (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data);

/**
 * @brief Unsubscribe a message to a given topic.
 */
int nns_edge_unsubscribe (nns_edge_h edge_h);

/**
 * @brief Get the topic of edge handle. Caller should release returned string using free().
 */
int nns_edge_get_topic (nns_edge_h edge_h, char **topic);

/**
 * @brief Set nnstreamer edge info.
 */
int nns_edge_set_info (nns_edge_h edge_h, const char *key, const char *value);

/**
 * @brief Get the nnstreamer edge event type.
 */
int nns_edge_event_get_type (nns_edge_event_h event_h, nns_edge_event_e *event);

/**
 * @brief Parse edge event (NNS_EDGE_EVENT_NEW_DATA_RECEIVED) and get received data.
 * @note Caller should release returned edge data using nns_edge_data_destroy().
 */
int nns_edge_event_parse_new_data (nns_edge_event_h event_h, nns_edge_data_h *data_h);

/**
 * @brief Parse edge event (NNS_EDGE_EVENT_CAPABILITY) and get capability string.
 * @note Caller should release returned string using free().
 */
int nns_edge_event_parse_capability (nns_edge_event_h event_h, char **capability);

/**
 * @brief Create nnstreamer edge data.
 */
int nns_edge_data_create (nns_edge_data_h *data_h);

/**
 * @brief Destroy nnstreamer edge data.
 */
int nns_edge_data_destroy (nns_edge_data_h data_h);

/**
 * @brief Validate edge data handle.
 */
int nns_edge_data_is_valid (nns_edge_data_h data_h);

/**
 * @brief Copy edge data and return new handle.
 */
int nns_edge_data_copy (nns_edge_data_h data_h, nns_edge_data_h *new_data_h);

/**
 * @brief Add raw data into nnstreamer edge data.
 */
int nns_edge_data_add (nns_edge_data_h data_h, void *data, size_t data_len, nns_edge_data_destroy_cb destroy_cb);

/**
 * @brief Get the nnstreamer edge data.
 * @note DO NOT release returned data. You should copy the data to another buffer if the returned data is necessary.
 */
int nns_edge_data_get (nns_edge_data_h data_h, unsigned int index, void **data, size_t *data_len);

/**
 * @brief Get the number of nnstreamer edge data.
 */
int nns_edge_data_get_count (nns_edge_data_h data_h, unsigned int *count);

/**
 * @brief Set the information of edge data.
 */
int nns_edge_data_set_info (nns_edge_data_h data_h, const char *key, const char *value);

/**
 * @brief Get the information of edge data. Caller should release the returned value using free().
 */
int nns_edge_data_get_info (nns_edge_data_h data_h, const char *key, char **value);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __NNSTREAMER_EDGE_H__ */
