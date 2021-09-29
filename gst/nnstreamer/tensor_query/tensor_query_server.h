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

/**
 * @brief set sink config
 */
void gst_tensor_query_server_set_sink_config (GstTensorsConfig *config);

/**
 * @brief get sink config
 */
void gst_tensor_query_server_get_sink_config (GstTensorsConfig *config);

/**
 * @brief set sink host
 */
void
gst_tensor_query_server_set_sink_host (gchar *host);

/**
 * @brief get sink host
 */
gchar *
gst_tensor_query_server_get_sink_host (void);

/**
 * @brief free sink host
 */
void
gst_tensor_query_server_free_sink_host (void);

/**
 * @brief set sink port
 */
void
gst_tensor_query_server_set_sink_port (guint16 port);

/**
 * @brief get sink port
 */
guint16
gst_tensor_query_server_get_sink_port (void);

G_END_DECLS

#endif /* __GST_TENSOR_QUERY_CLIENT_H__ */
