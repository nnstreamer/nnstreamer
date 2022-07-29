/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_server.h
 * @date    03 Aug 2021
 * @brief   GStreamer plugin to handle meta_query for server elements
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_TENSOR_QUERY_SERVER_H__
#define __GST_TENSOR_QUERY_SERVER_H__

#include <gst/gst.h>
#include <tensor_common.h>
#include "nnstreamer-edge.h"
#include "tensor_meta.h"

G_BEGIN_DECLS

#define DEFAULT_SERVER_ID 0
#define DEFAULT_QUERY_INFO_TIMEOUT 5
typedef void * edge_server_handle;

/**
 * @brief GstTensorQueryServer internal info data structure.
 */
typedef struct
{
  char *id;
  gboolean configured;
  GMutex lock;
  GCond cond;

  nns_edge_h edge_h;
} GstTensorQueryServer;

/**
 * @brief Get nnstreamer edge server handle.
 */
edge_server_handle
gst_tensor_query_server_get_handle (char *id);

/**
 * @brief Add GstTensorQueryServer.
 */
edge_server_handle
gst_tensor_query_server_add_data (char *id, nns_edge_connect_type_e connect_type);

/**
 * @brief Remove GstTensorQueryServer.
 */
void
gst_tensor_query_server_remove_data (edge_server_handle server_h);

/**
 * @brief Wait until the sink is configured and get server info handle.
 */
gboolean
gst_tensor_query_server_wait_sink (edge_server_handle server_h);

/**
 * @brief Get edge handle from server data.
 */
nns_edge_h
gst_tensor_query_server_get_edge_handle (edge_server_handle server_h);

/**
 * @brief set query server sink configured.
 */
void
gst_tensor_query_server_set_configured (edge_server_handle server_h);

G_END_DECLS

#endif /* __GST_TENSOR_QUERY_CLIENT_H__ */
