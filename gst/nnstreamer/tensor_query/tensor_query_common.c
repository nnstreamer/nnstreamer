/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.c
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @see    gsttcpserversink, gsttcpserversrc
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "nnstreamer_log.h"
#include "tensor_query_common.h"

/** @todo change max read size or need payload */
#define MAX_READ_SIZE 4096

/**
 * @brief Create requested socket.
 * @param[in] hostname the hostname.
 * @param[in] port a port number.
 * @param[in] cancellable (nullable) GCancellable
 * @param[out] saddr Socket address
 * @return Newly created socket or NULL on error.
 * @note Caller is responsible for unreferring the returned object with g_object_unref().
 */
GSocket *
gst_tensor_query_socket_new (const gchar * hostname, guint16 port,
    GCancellable * cancellable, GSocketAddress ** saddr)
{
  GSocket *socket;
  GError *err = NULL;
  GInetAddress *addr;

  /* look up name if we need to */
  addr = g_inet_address_new_from_string (hostname);
  if (!addr) {
    GList *results;
    GResolver *resolver;

    resolver = g_resolver_get_default ();
    results = g_resolver_lookup_by_name (resolver, hostname, cancellable, &err);

    if (!results) {
      if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
        nns_logd ("gst_tensor_query_socket_new: Cancelled name resolval");
      } else {
        nns_loge ("Failed to resolve host '%s': %s", hostname, err->message);
      }
      g_clear_error (&err);
      g_object_unref (resolver);
      return NULL;
    }
    /** @todo Try with the second address if the first fails */
    addr = G_INET_ADDRESS (g_object_ref (results->data));

    g_resolver_free_addresses (results);
    g_object_unref (resolver);
  }

  *saddr = g_inet_socket_address_new (addr, port);
  g_object_unref (addr);

  /* create sending client socket */
  /** @todo Support UDP protocol */
  socket =
      g_socket_new (g_socket_address_get_family (*saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);

  return socket;
}

/**
 * @brief Receive data from a socket.
 * @param[in] socket the socket.
 * @param[in] cancellable (nullable) GCancellable.
 * @param[in/out] bytes_received Add the number of bytes received.
 * @param[out] outbuf output buffer filled by reveived data.
 * @return GST_FLOW_OK if there is no error.
 */
GstFlowReturn
gst_tensor_query_socket_receive (GSocket * socket, GCancellable * cancellable,
    gsize * bytes_received, GstBuffer * outbuf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  gssize rret, avail;
  gsize read;
  GError *err = NULL;
  GstMapInfo map;
  GstMemory *out_mem;
  gboolean need_alloc;

  /* read the buffer header */
  avail = g_socket_get_available_bytes (socket);
  if (avail < 0) {
    nns_loge ("Failed to get available bytes from socket");
    ret = GST_FLOW_ERROR;
    goto done;
  } else if (avail == 0) {
    GIOCondition condition;

    if (!g_socket_condition_wait (socket,
            G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP, cancellable, &err))
      goto select_error;

    condition =
        g_socket_condition_check (socket,
        G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP);

    if ((condition & G_IO_ERR)) {
      nns_loge ("Socket in error state");
      outbuf = NULL;
      ret = GST_FLOW_ERROR;
      goto done;
    } else if ((condition & G_IO_HUP)) {
      nns_logd ("Connection closed");
      outbuf = NULL;
      ret = GST_FLOW_EOS;
      goto done;
    }
    avail = g_socket_get_available_bytes (socket);
    if (avail < 0) {
      nns_loge ("Failed to get available bytes from socket");
      ret = GST_FLOW_ERROR;
      goto done;
    }
  }

  if (avail > 0) {
    nns_logi ("available data: %d", (gint) avail);
    read = MIN (avail, MAX_READ_SIZE);
    need_alloc = (gst_buffer_get_size (outbuf) == 0);

    if (need_alloc) {
      out_mem = gst_allocator_alloc (NULL, read, NULL);
      nns_logi ("allocated buffer size: %lu",
          gst_memory_get_sizes (out_mem, NULL, NULL));
    } else {
      if (gst_buffer_get_size (outbuf) < read) {
        gst_buffer_set_size (outbuf, read);
      }
      out_mem = gst_buffer_get_all_memory (outbuf);
    }

    gst_memory_map (out_mem, &map, GST_MAP_READWRITE);
    rret =
        g_socket_receive (socket, (gchar *) map.data, read, cancellable, &err);
    gst_memory_unmap (out_mem, &map);

    if (need_alloc)
      gst_buffer_append_memory (outbuf, out_mem);
    else
      gst_memory_unref (out_mem);
  } else {
    /* Connection closed */
    rret = 0;
    outbuf = NULL;
    read = 0;
  }

  if (rret == 0) {
    nns_logd ("Connection closed");
    ret = GST_FLOW_EOS;
    if (outbuf) {
      gst_buffer_unref (outbuf);
    }
    outbuf = NULL;
  } else if (rret < 0) {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      ret = GST_FLOW_FLUSHING;
      nns_logd ("Cancelled reading from socket");
    } else {
      ret = GST_FLOW_ERROR;
      nns_loge ("Failed to read from socket: %s", err->message);
    }
    gst_buffer_unref (outbuf);
    outbuf = NULL;
  } else {
    ret = GST_FLOW_OK;
    *bytes_received += read;
  }
  g_clear_error (&err);

done:
  return ret;

select_error:
  {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      nns_logd ("Cancelled select");
      ret = GST_FLOW_FLUSHING;
    } else {
      nns_loge ("Select failed: %s", err->message);
      ret = GST_FLOW_ERROR;
    }
    g_clear_error (&err);
    return ret;
  }
}
