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
#include <nnstreamer-edge.h>
#include "tensor_meta.h"

G_BEGIN_DECLS
#define DEFAULT_SERVER_ID 0
#define DEFAULT_QUERY_INFO_TIMEOUT 5

/**
 * @brief Internal data structure for nns-edge info to prepare edge connection.
 */
typedef struct
{
  gchar *host;
  guint16 port;
  gchar *dest_host;
  guint16 dest_port;
  gchar *topic;

  /* nns-edge callback info */
  nns_edge_event_cb cb;
  void *pdata;
} GstTensorQueryEdgeInfo;

/**
 * @brief GstTensorQueryServer internal info data structure.
 */
typedef struct
{
  guint id;
  gboolean configured;
  GMutex lock;
  GCond cond;

  nns_edge_h edge_h;
} GstTensorQueryServer;

/**
 * @brief Add GstTensorQueryServer.
 */
gboolean gst_tensor_query_server_add_data (const guint id);

/**
 * @brief Prepare edge connection and its handle.
 */
gboolean gst_tensor_query_server_prepare (const guint id,
    nns_edge_connect_type_e connect_type, GstTensorQueryEdgeInfo *edge_info);

/**
 * @brief Remove GstTensorQueryServer.
 */
void gst_tensor_query_server_remove_data (const guint id);

/**
 * @brief Wait until the sink is configured and get server info handle.
 */
gboolean gst_tensor_query_server_wait_sink (const guint id);

/**
 * @brief Send buffer to connected edge device.
 */
gboolean gst_tensor_query_server_send_buffer (const guint id, GstBuffer *buffer);

/**
 * @brief set query server sink configured.
 */
void gst_tensor_query_server_set_configured (const guint id);

/**
 * @brief set query server caps.
 */
void gst_tensor_query_server_set_caps (const guint id, const gchar *caps_str);

/**
 * @brief Release nnstreamer edge handle of query server.
 */
void gst_tensor_query_server_release_edge_handle (const guint id);

G_END_DECLS
#endif /* __GST_TENSOR_QUERY_CLIENT_H__ */
