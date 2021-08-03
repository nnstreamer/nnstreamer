/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.c
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @author Junhwan Kim <jejudo.kim@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gio/gio.h>
#include <gio/gsocket.h>
#include <stdint.h>
#include <errno.h>
#include <nnstreamer_util.h>
#include <nnstreamer_log.h>
#include "tensor_query_common.h"

#define TENSOR_QUERY_SERVER_DATA_LEN 128
#define TCP_BACKLOG 1

/**
 * @brief Structures for tensor query server data.
 */
typedef struct
{
  TensorQueryProtocol protocol;
  gchar *server_host;
  guint32 server_port;
  /* network info */
  union
  {
    struct
    {
      GSocket *server_socket;
      GCancellable *cancellable;
    };
    // check the size of struct is less
    guint8 _dummy[TENSOR_QUERY_SERVER_DATA_LEN];
  };
} TensorQueryServerData;

/**
 * @brief Structures for tensor query client data.
 */
typedef struct
{
  TensorQueryProtocol protocol;
  union
  {
    struct
    {
      GSocket *client_socket;
      GCancellable *cancellable;
    };
  };
} TensorQueryClientData;

/**
 * @brief Connection info structure
 */
typedef struct
{
  TensorQueryProtocol protocol;
  gchar *host;
  guint32 port;
  /* network info */
  union
  {
    /* TCP */
    struct
    {
      GSocket *socket;
      GCancellable *cancellable;
    };
  };
} TensorQueryConnection;

static GSocket *query_establish_socket (const char *host, uint32_t port);
static gboolean
query_tcp_receive (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable);
static gboolean query_tcp_send (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable);

/**
 * @brief Create requested socket.
 */
static gboolean
gst_tensor_query_socket_new (query_connection_handle conn_h,
    GSocketAddress ** saddr)
{
  GError *err = NULL;
  GInetAddress *addr;
  TensorQueryConnection *conn = (TensorQueryConnection *) conn_h;

  /* look up name if we need to */
  addr = g_inet_address_new_from_string (conn->host);
  if (!addr) {
    GList *results;
    GResolver *resolver;
    resolver = g_resolver_get_default ();
    results =
        g_resolver_lookup_by_name (resolver, conn->host, conn->cancellable,
        &err);
    if (!results) {
      if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
        nns_logd ("gst_tensor_query_socket_new: Cancelled name resolval");
      } else {
        nns_loge ("Failed to resolve host '%s': %s", conn->host, err->message);
      }
      g_clear_error (&err);
      g_object_unref (resolver);
      return FALSE;
    }
    /** @todo Try with the second address if the first fails */
    addr = G_INET_ADDRESS (g_object_ref (results->data));
    g_resolver_free_addresses (results);
    g_object_unref (resolver);
  }

  *saddr = g_inet_socket_address_new (addr, conn->port);
  g_object_unref (addr);

  /* create sending client socket */
  /** @todo Support UDP protocol */
  conn->socket =
      g_socket_new (g_socket_address_get_family (*saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);
  if (!conn->socket) {
    nns_loge ("Failed to create new socket");
    return FALSE;
  }
  return TRUE;
}

/**
 * @brief Connect to the specified address.
 */
query_connection_handle
nnstreamer_query_connect (TensorQueryProtocol protocol, const char *ip,
    uint32_t port, uint32_t timeout_ms)
{
  /** @todo remove "UNUSED" when you implement the full features */
  TensorQueryConnection *conn = g_new0 (TensorQueryConnection, 1);
  UNUSED (timeout_ms);

  conn->protocol = protocol;
  conn->host = g_strdup (ip);
  conn->port = port;
  switch (protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      GError *err = NULL;
      GSocketAddress *saddr = NULL;

      conn->cancellable = g_cancellable_new ();
      if (!gst_tensor_query_socket_new (conn, &saddr)) {
        nns_loge ("Failed to create new socket");
        goto tcp_fail;
      }

      if (!g_socket_connect (conn->socket, saddr, conn->cancellable, &err)) {
        if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
          nns_logd ("Cancelled connecting");
        } else {
          nns_loge ("Failed to connect to host");
        }
        goto tcp_fail;
      }
      g_object_unref (saddr);
      break;
    tcp_fail:
      nnstreamer_query_close (conn);
      g_object_unref (saddr);
      g_error_free (err);
      return NULL;
    }
    default:
      nns_loge ("Unsupported protocol.");
      return NULL;
  }
  return conn;
}

/**
 * @brief receive command from connected device.
 * @return 0 if OK, negative value if error
 */
int
nnstreamer_query_receive (query_connection_handle connection,
    TensorQueryCommandData * data, uint32_t timeout_ms)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;
  UNUSED (timeout_ms);
  if (!conn) {
    nns_loge ("Invalid connection data");
    return -EINVAL;
  }
  data->protocol = conn->protocol;
  switch (conn->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      TensorQueryCommand cmd;

      if (!query_tcp_receive (conn->socket, (uint8_t *) & cmd, sizeof (cmd),
              conn->cancellable)) {
        nns_loge ("Failed to receive from socket");
        return -EIO;
      }
      data->cmd = cmd;

      if (cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA) {
        /* receive data */
        if (!query_tcp_receive (conn->socket, (uint8_t *) data->data.data,
                data->data.size, conn->cancellable)) {
          nns_loge ("Failed to receive from socket");
          return -EIO;
        }
        return 0;
      } else {
        /* receive data_info */
        if (!query_tcp_receive (conn->socket, (uint8_t *) & data->data_info,
                sizeof (TensorQueryDataInfo), conn->cancellable)) {
          nns_loge ("Failed to receive from socket");
          return -EIO;
        }
      }
    }
      break;
    default:
      /* NYI */
      return -ENOSYS;
  }
  return 0;
}

/**
 * @brief send command to connected device.
 * @return 0 if OK, negative value if error
 */
int
nnstreamer_query_send (query_connection_handle connection,
    TensorQueryCommandData * data, uint32_t timeout_ms)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;
  UNUSED (timeout_ms);
  if (!data) {
    nns_loge ("Sending data is NULL");
    return -EINVAL;
  }
  if (!conn) {
    nns_loge ("Invalid connection data");
    return -EINVAL;
  }

  switch (conn->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
      if (!query_tcp_send (conn->socket, (uint8_t *) & data->cmd,
              sizeof (TensorQueryCommand), conn->cancellable)) {
        nns_loge ("Failed to send to socket");
        return -EIO;
      }
      if (data->cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA) {
        /* send data */
        if (!query_tcp_send (conn->socket, (uint8_t *) data->data.data,
                data->data.size, conn->cancellable)) {
          nns_loge ("Failed to send to socket");
          return -EIO;
        }
      } else {
        /* send data_info */
        if (!query_tcp_send (conn->socket, (uint8_t *) & data->data_info,
                sizeof (TensorQueryDataInfo), conn->cancellable)) {
          nns_loge ("Failed to send to socket");
          return -EIO;
        }
      }
      break;
    default:
      /* NYI */
      return -ENOSYS;
  }
  return 0;
}

/**
 * @brief free connection
 * @return 0 if OK, negative value if error
 */
int
nnstreamer_query_close (query_connection_handle connection)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;
  if (!conn) {
    nns_loge ("Invalid connection data");
    return -EINVAL;
  }
  switch (conn->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      GError *err;
      if (!g_socket_close (conn->socket, &err)) {
        nns_loge ("Failed to close socket: %s", err->message);
        g_error_free (err);
        return -EIO;
      }
      g_object_unref (conn->socket);
      g_object_unref (conn->cancellable);
    }
      break;
    default:
      /* NYI */
      return -ENOSYS;
  }
  g_free (conn);
  return 0;
}

/**
 * @brief return initialized server handle
 * @return query_server_handle, NULL if error
 */
query_server_handle
nnstreamer_query_server_data_new (void)
{
  TensorQueryServerData *sdata = g_try_new (TensorQueryServerData, 1);
  if (!sdata) {
    nns_loge ("Failed to allocate server data");
    return NULL;
  }
  sdata->cancellable = g_cancellable_new ();
  sdata->protocol = _TENSOR_QUERY_PROTOCOL_END;
  /* init union */
  memset (sdata->_dummy, 0, sizeof (sdata->_dummy));
  return (query_server_handle) sdata;
}

/**
 * @brief free server handle
 */
void
nnstreamer_query_server_data_free (query_server_handle server_data)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  GError *err;
  if (!sdata)
    return;

  switch (sdata->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
      if (sdata->server_socket) {
        if (!g_socket_close (sdata->server_socket, &err)) {
          nns_logw ("Failed to close socket: %s", err->message);
          g_error_free (err);
        }
        g_object_unref (sdata->server_socket);
      }
      g_object_unref (sdata->cancellable);
      break;
    default:
      /* NYI */
      nns_logw ("Invalid protocol");
      break;
  }
  g_free (sdata->server_host);
  g_free (sdata);
}

/**
 * @brief set server handle params and setup server
 * @return 0 if OK, negative value if error 
 */
int
nnstreamer_query_server_data_setup (query_server_handle server_data,
    TensorQueryProtocol protocol, const char *host, uint32_t port)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  if (!sdata)
    return -EINVAL;

  sdata->protocol = protocol;
  sdata->server_host = g_strdup (host);
  sdata->server_port = port;

  switch (protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
      sdata->server_socket = query_establish_socket (host, port);
      if (!sdata->server_socket) {
        nns_loge ("Failed establish socket");
        return -EIO;
      }
      break;
    default:
      /* NYI */
      return -ENOSYS;
  }
  return 0;
}

/**
 * @brief accept client connection
 */
query_connection_handle
nnstreamer_query_server_accept (query_server_handle server_data)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  TensorQueryConnection *conn;
  GError *err;

  switch (sdata->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      GSocket *client_socket;
      GSocketAddress *saddr;
      client_socket = g_socket_accept (sdata->server_socket,
          sdata->cancellable, &err);
      if (!client_socket) {
        nns_loge ("Failed to accept client: %s", err->message);
        g_error_free (err);
        return 0;
      }
      saddr = g_socket_get_remote_address (client_socket, &err);
      if (!saddr) {
        nns_loge ("Failed to get client address: %s", err->message);
        g_error_free (err);
        return 0;
      }

      conn = g_new0 (TensorQueryConnection, 1);
      conn->protocol = sdata->protocol;
      conn->host =
          g_inet_address_to_string (g_inet_socket_address_get_address (
              (GInetSocketAddress *) saddr));
      conn->port =
          g_inet_socket_address_get_port ((GInetSocketAddress *) saddr);
      conn->socket = client_socket;
      conn->cancellable = g_cancellable_new ();

      nns_logi ("Inserted client %s:%d", conn->host, conn->port);
      g_object_unref (saddr);
    }
      break;
    default:
      /* NYI */
      return 0;
  }
  return conn;
}

/**
 * @brief [TCP] establish socket for tcp server
 */
static GSocket *
query_establish_socket (const char *host, uint32_t port)
{
  GSocket *server_socket;
  GSocketAddress *saddr;
  GError *err;

  saddr = g_inet_socket_address_new_from_string (host, port);
  if (!saddr) {
    /* hostname resolution not supported */
    nns_loge ("Failed to parse host `%s:%u`", host, port);
    goto fail;
  }

  server_socket =
      g_socket_new (g_socket_address_get_family (saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);

  if (!server_socket) {
    nns_loge ("Failed to create socket : %s", err->message);
    goto fail;
  }

  if (!g_socket_bind (server_socket, saddr, TRUE, &err)) {
    nns_loge ("Failed to bind on host `%s:%u`: %s", host, port, err->message);
    goto fail;
  }

  g_socket_set_listen_backlog (server_socket, TCP_BACKLOG);
  /* listen */
  if (!g_socket_listen (server_socket, &err)) {
    nns_loge ("Failed to listen on host `%s:%u`: %s", host, port, err->message);
    goto close_socket;
  }
  nns_logd ("Listening on port %d",
      g_inet_socket_address_get_port ((GInetSocketAddress *) saddr));
  g_object_unref (saddr);
  return server_socket;

fail:
  g_object_unref (saddr);
  g_error_free (err);
  return NULL;

close_socket:
  if (!g_socket_close (server_socket, &err)) {
    nns_loge ("Failed to close socket: %s", err->message);
  }
  g_object_unref (server_socket);
  g_object_unref (saddr);
  g_error_free (err);
  return NULL;
}

/**
 * @brief [TCP] receive data for tcp server
 */
static gboolean
query_tcp_receive (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable)
{
  gsize bytes_received = 0;
  gssize rret;
  GError *err;

  while (bytes_received < size) {
    rret = g_socket_receive (socket, (gchar *) data + bytes_received,
        size - bytes_received, cancellable, &err);
    if (rret < 0) {
      nns_loge ("Failed to read from socket: %s", err->message);
      g_clear_error (&err);
      return FALSE;
    }
    bytes_received += rret;
  }
  nns_logd ("received %zu", bytes_received);
  return TRUE;
}

/**
 * @brief [TCP] send data for tcp server
 */
static gboolean
query_tcp_send (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable)
{
  gsize bytes_sent = 0;
  gssize rret;
  GError *err;
  while (bytes_sent < size) {
    rret = g_socket_send (socket, (gchar *) data + bytes_sent,
        size - bytes_sent, cancellable, &err);
    if (rret < 0) {
      nns_loge ("Error while sending data %s", err->message);
      g_clear_error (&err);
      return FALSE;
    }
    bytes_sent += rret;
  }
  nns_logd ("sent %zu", bytes_sent);
  return TRUE;
}
