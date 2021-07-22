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

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

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
  _TENSOR_QUERY_CMD_TRANSFER_START = 0,
  _TENSOR_QUERY_CMD_TRANSFER_DATA = 1,
  _TENSOR_QUERY_CMD_TRANSFER_END = 2,
  _TENSOR_QUERY_CMD_END
} TensorQueryCommand;

/**
 * @brief Structures for tensor query data info.
 */
typedef struct
{
  GstTensorsConfig config;
  int64_t base_time;
  int64_t sent_time;
  uint64_t duration;
  uint64_t dts;
  uint64_t pts;
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
  union
  {
    TensorQueryDataInfo data_info; /** _TENSOR_QUERY_CMD_TRANSFER_START */
    TensorQueryData data;          /** _TENSOR_QUERY_CMD_TRANSFER_DATA */
  };
} TensorQueryCommandData;

/**
 * @brief generate unique id.
 * @return unique id if OK, 0 if error
 */
extern uint64_t
nnstreamer_query_request_id (const char *ip, uint32_t port, int is_client);

/**
 * @brief connect to the specified address.
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_connect (uint64_t id, const char *ip, uint32_t port, uint32_t timeout_ms);

/**
 * @brief send command to connected device.
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_send (uint64_t id, TensorQueryCommandData *data, uint32_t timeout_ms);

/**
 * @brief receive command from connected device.
 * @return 0 if OK, negative value if error
 */
extern int
nnstreamer_query_receive (uint64_t id, TensorQueryCommandData *data, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TENSOR_QUERY_COMMON_H__ */
