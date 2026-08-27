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
#define GST_EDGE_PACKAGE "edge"
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

/**
 * @brief Parse custom properties and set them on the edge handle.
 * @param[in] edge_h The edge handle to set the parsed options on.
 * @param[in] custom_props Comma-separated key:value pairs, e.g. "key1:val1,key2:val2".
 *            A value may contain ':' (e.g. "QUEUE_SIZE:10:OLD"); only the first
 *            ':' separates the key from the value. Whitespace around keys and
 *            values is trimmed.
 * @return TRUE if every token was applied. Empty or whitespace-only tokens
 *         (e.g. from a trailing comma) are silently ignored and do not affect
 *         the result. Tokens without ':', with an empty key or value, or
 *         rejected by nns_edge_set_info() are logged, skipped, and make the
 *         return value FALSE. Option values may carry credentials of a custom
 *         connection library, so a rejected option is named by its key, or by
 *         its position when no key could be parsed out of it.
 */
gboolean gst_edge_parse_custom_props (nns_edge_h edge_h, const gchar * custom_props);

G_END_DECLS
#endif /* __GST_EDGE_H__ */
