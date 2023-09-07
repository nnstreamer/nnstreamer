/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_common.h
 * @date    01 Aug 2022
 * @brief   Common functions for edge sink and src
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_EDGE_H__
#define __GST_EDGE_H__

#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer-edge.h>

#ifndef GST_EDGE_PACKAGE
#define GST_EDGE_PACKAGE "GStreamer Edge Plugins"
#endif /* GST_EDGE_PACKAGE */
#define GST_EDGE_ELEM_NAME_SINK "edgesink"
#define GST_EDGE_ELEM_NAME_SRC "edgesrc"
#define DEFAULT_HOST "localhost"
#define DEFAULT_PORT 3000
#define DEFAULT_CONNECT_TYPE (NNS_EDGE_CONNECT_TYPE_TCP)
#define GST_TYPE_EDGE_CONNECT_TYPE (gst_edge_get_connect_type ())

G_BEGIN_DECLS

/**
 * @brief register GEnumValue array for edge protocol property handling
 */
GType gst_edge_get_connect_type (void);

G_END_DECLS
#endif /* __GST_EDGE_H__ */
