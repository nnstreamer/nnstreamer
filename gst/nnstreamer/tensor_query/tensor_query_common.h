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
 * @see    gsttcpserversink, gsttcpserversrc
 */

#ifndef __GST_TENSOR_QUERY_COMMON_H__
#define __GST_TENSOR_QUERY_COMMON_H__

#include <glib.h>
#include <gst/gst.h>
#include <gio/gio.h>
#include <gio/gsocket.h>

G_BEGIN_DECLS

/**
 * @brief protocol options for tensor query
 */
typedef enum
{
  QUERY_PROTOCOL_TCP = 0,
  QUERY_PROTOCOL_UDP = 1,
  QUERY_PROTOCOL_MQTT = 2,
  QUERY_PROTOCOL_END,
} tensor_query_protocol;

/**
 * @brief Create requested socket.
 * @param[in] hostname the hostname
 * @param[in] port a port number
 * @param[in] cancellable (nullable) GCancellable
 * @return Newly created socket or NULL on error.
 * @note Caller is responsible for unreffiring the returned object with g_object_unref().
 */
extern GSocket *
gst_tensor_query_socket_new (const gchar * hostname, guint16 port,
    GCancellable * cancellable);
/**
 * @brief Receive data from a socket.
 * @param[in] socket the socket.
 * @param[in] cancellable (nullable) GCancellable
 * @param[in/out] bytes_received Add the number of received bytes to bytes_received.
 * @param[out] outbuf output buffer filled by reveived data.
 * @return GST_FLOW_OK if there is no error.
 */
extern GstFlowReturn
gst_tensor_query_socket_receive (GSocket * socket, GCancellable * cancellable,
    gsize * bytes_received, GstBuffer * outbuf);

G_END_DECLS
#endif /* __GST_TENSOR_QUERY_COMMON_H__ */
