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

G_BEGIN_DECLS

#define DEFAULT_SERVER_ID 0
#define DEFAULT_QUERY_INFO_TIMEOUT 5
typedef void * query_server_info_handle;

/**
 * @brief GstTensorQueryServer internal info data structure.
 */
typedef struct
{
  gint64 id;
  GstTensorsConfig sink_config;
  gchar *sink_host;
  guint16 sink_port;
  gboolean configured;
  GMutex lock;
  GCond cond;
} GstTensorQueryServerInfo;

/**
 * @brief Add GstTensorQueryServerInfo.
 */
query_server_info_handle
gst_tensor_query_server_add_data (guint id);

/**
 * @brief Remove GstTensorQueryServerInfo.
 */
void
gst_tensor_query_server_remove_data (query_server_info_handle server_info_h);

/**
 * @brief Wait until the sink is configured and get server info handle.
 */
gboolean
gst_tensor_query_server_wait_sink (query_server_info_handle server_info_h);

/**
 * @brief set sink config
 */
void gst_tensor_query_server_set_sink_config (query_server_info_handle server_info_h, GstTensorsConfig *config);

/**
 * @brief get sink config
 */
void gst_tensor_query_server_get_sink_config (query_server_info_handle server_info_h, GstTensorsConfig *config);

/**
 * @brief set sink host address and port
 */
void
gst_tensor_query_server_set_sink_host (query_server_info_handle server_info_h, gchar *host, guint16 port);

/**
 * @brief get sink host
 */
gchar *
gst_tensor_query_server_get_sink_host (query_server_info_handle server_info_h);

/**
 * @brief get sink port
 */
guint16
gst_tensor_query_server_get_sink_port (query_server_info_handle server_info_h);

G_END_DECLS

#endif /* __GST_TENSOR_QUERY_CLIENT_H__ */
