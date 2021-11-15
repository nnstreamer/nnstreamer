/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.h
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __TENSOR_QUERY_COMMON_H__
#define __TENSOR_QUERY_COMMON_H__

#include "tensor_typedef.h"
#include "tensor_common.h"
#include "tensor_meta.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void * query_connection_handle;
typedef void * query_server_handle;
#define DEFAULT_TIMEOUT_MS 100000

/**
 * @brief protocol options for tensor query.
 */
typedef enum
{
  _TENSOR_QUERY_PROTOCOL_TCP = 0,
  _TENSOR_QUERY_PROTOCOL_UDP = 1,
  _TENSOR_QUERY_PROTOCOL_MQTT = 2,
  _TENSOR_QUERY_PROTOCOL_END
} TensorQueryProtocol;

/**
 * @brief Structures for tensor query commands.
 */
typedef enum
{
  _TENSOR_QUERY_CMD_REQUEST_INFO = 0,
  _TENSOR_QUERY_CMD_RESPOND_APPROVE = 1,
  _TENSOR_QUERY_CMD_RESPOND_DENY = 2,
  _TENSOR_QUERY_CMD_TRANSFER_START = 3,
  _TENSOR_QUERY_CMD_TRANSFER_DATA = 4,
  _TENSOR_QUERY_CMD_TRANSFER_END = 5,
  _TENSOR_QUERY_CMD_CLIENT_ID = 6,
  _TENSOR_QUERY_CMD_END
} TensorQueryCommand;


/**
 * @brief Structures for tensor query data info.
 */
typedef struct
{
  int64_t base_time;
  int64_t sent_time;
  uint64_t duration;
  uint64_t dts;
  uint64_t pts;
  uint32_t num_mems;
  uint64_t mem_sizes[NNS_TENSOR_SIZE_LIMIT];
} TensorQueryDataInfo;

typedef struct
{
  uint8_t *data;
  size_t size;
} TensorQueryData;

/**
 * @brief Structures for tensor query command buffers.
 */
typedef struct
{
  TensorQueryCommand cmd;
  TensorQueryProtocol protocol;
  query_client_id_t client_id;
  union
  {
    TensorQueryDataInfo data_info; /** _TENSOR_QUERY_CMD_TRANSFER_START */
    TensorQueryData data;          /** _TENSOR_QUERY_CMD_TRANSFER_DATA */
  };
} TensorQueryCommandData;

/**
 * @brief connect to the specified address.
 * @return 0 if OK, negative value if error
 */
extern query_connection_handle
nnstreamer_query_connect (TensorQueryProtocol protocol, const char *ip, uint16_t port, uint32_t timeout_ms);

/**
 * @brief get client id from query connection handle
 */
extern query_client_id_t
nnstreamer_query_connection_get_client_id (query_connection_handle connection);

/**
 * @brief get port from query connection handle
 */
extern uint16_t
nnstreamer_query_connection_get_port (query_connection_handle connection);

/**
 * @brief send command to connected device.
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_send (query_connection_handle connection, TensorQueryCommandData *data, uint32_t timeout_ms);

/**
 * @brief receive command from connected device.
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_receive (query_connection_handle connection, TensorQueryCommandData *data);

/**
 * @brief close connection with corresponding id.
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_close (query_connection_handle connection);

/* server */
/**
 * @brief accept connection from remote
 * @return query_connection_handle including connection data
 */
extern query_connection_handle
nnstreamer_query_server_accept (query_server_handle server_data);

/**
 * @brief return initialized server handle
 * @return query_server_handle, NULL if error
 */
extern query_server_handle
nnstreamer_query_server_data_new (void);

/**
 * @brief free server handle
 */
extern void
nnstreamer_query_server_data_free (query_server_handle server_data);

/**
 * @brief set server handle params and setup server
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_server_init (query_server_handle server_data,
    TensorQueryProtocol protocol, const char *host, uint16_t port, int8_t is_src);

/**
 * @brief set server source and sink tensors config.
 */
extern void
nnstreamer_query_server_data_set_caps_str (query_server_handle server_data,
    const char * src_caps_str, const char * sink_caps_str);

/**
 * @brief Get buffer from message queue.
 */
extern GstBuffer *
nnstreamer_query_server_get_buffer (query_server_handle server_data);

/**
 * @brief Send gst-buffer to destination node.
 * @return True if all data in gst-buffer is successfully sent. False if failed to transfer data.
 * @todo This function should be used in nnstreamer element. Update function name rule and params later.
 */
extern gboolean
tensor_query_send_buffer (query_connection_handle connection,
    GstElement * element, GstBuffer * buffer, guint timeout);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TENSOR_QUERY_COMMON_H__ */
