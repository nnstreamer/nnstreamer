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

G_END_DECLS

#endif /* __GST_TENSOR_QUERY_CLIENT_H__ */
