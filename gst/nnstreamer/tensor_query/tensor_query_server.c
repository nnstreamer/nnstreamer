/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_server.c
 * @date    03 Aug 2021
 * @brief   GStreamer plugin to handle meta_query for server elements
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "tensor_query_server.h"
#include <nnstreamer_util.h>
#include <tensor_typedef.h>
#include <tensor_common.h>

/**
 * @brief sink config is shared for src
 */
static GstTensorsConfig sink_config;
static gchar *sink_host;
static guint16 sink_port;

/**
 * @brief set sink config
 */
void
gst_tensor_query_server_set_sink_config (GstTensorsConfig * config)
{
  gst_tensors_config_copy (&sink_config, config);
}

/**
 * @brief get sink config
 */
void
gst_tensor_query_server_get_sink_config (GstTensorsConfig * config)
{
  gst_tensors_config_copy (config, &sink_config);
}

/**
 * @brief set sink host
 */
void
gst_tensor_query_server_set_sink_host (gchar * host)
{
  sink_host = g_strdup (host);
}

/**
 * @brief get sink host
 */
gchar *
gst_tensor_query_server_get_sink_host (void)
{
  return g_strdup (sink_host);
}

/**
 * @brief free sink host
 */
void
gst_tensor_query_server_free_sink_host (void)
{
  g_free (sink_host);
  sink_host = NULL;
}

/**
 * @brief set sink port
 */
void
gst_tensor_query_server_set_sink_port (guint16 port)
{
  sink_port = port;
}

/**
 * @brief get sink port
 */
guint16
gst_tensor_query_server_get_sink_port (void)
{
  return sink_port;
}
