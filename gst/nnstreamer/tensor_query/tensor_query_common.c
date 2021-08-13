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
#include <string.h>
#include <errno.h>
#include <nnstreamer_util.h>
#include <nnstreamer_log.h>
#include "tensor_query_common.h"

#define TENSOR_QUERY_SERVER_DATA_LEN 128
#define N_BACKLOG 10
#define CLIENT_ID_LEN 4

/**
 * @brief Query server dependent network data
 */
typedef struct
{
  TensorQueryProtocol protocol;
  int8_t is_src;
  union
  {
    struct
    {
      GSocketListener *socket_listener;
      GCancellable *cancellable;
      GAsyncQueue *conn_queue;
    };
    /* check the size of struct is less */
    uint8_t _dummy[TENSOR_QUERY_SERVER_DATA_LEN];
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
  char *host;
  uint16_t port;
  uint32_t client_id;
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

static gboolean
query_tcp_receive (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable);
static gboolean query_tcp_send (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable);
static void
accept_socket_async_cb (GObject * source, GAsyncResult * result,
    gpointer user_data);

/**
 * @brief get client id from query connection handle
 */
uint32_t
nnstreamer_query_connection_get_client_id (query_connection_handle connection)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;
  return conn->client_id;
}

/**
 * @brief get port from query connection handle
 */
uint32_t
nnstreamer_query_connection_get_port (query_connection_handle connection)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;
  return conn->port;
}


/**
 * @brief Get socket address
 */
static gboolean
gst_tensor_query_get_saddr (const char *host, uint16_t port,
    GCancellable * cancellable, GSocketAddress ** saddr)
{
  GError *err = NULL;
  GInetAddress *addr;

  /* look up name if we need to */
  addr = g_inet_address_new_from_string (host);
  if (!addr) {
    GList *results;
    GResolver *resolver;
    resolver = g_resolver_get_default ();
    results = g_resolver_lookup_by_name (resolver, host, cancellable, &err);
    if (!results) {
      if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
        nns_logd ("gst_tensor_query_socket_new: Cancelled name resolval");
      } else {
        nns_loge ("Failed to resolve host '%s': %s", host, err->message);
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

  *saddr = g_inet_socket_address_new (addr, port);
  g_object_unref (addr);

  return TRUE;
}

/**
 * @brief Create and connect to requested socket.
 */
static gboolean
gst_tensor_query_connect (query_connection_handle conn_h)
{
  GError *err = NULL;
  GSocketAddress *saddr = NULL;
  TensorQueryConnection *conn = (TensorQueryConnection *) conn_h;
  gboolean ret = FALSE;

  if (!gst_tensor_query_get_saddr (conn->host, conn->port, conn->cancellable,
          &saddr)) {
    nns_loge ("Failed to get socket address");
    return ret;
  }

  /* create sending client socket */
  /** @todo Support UDP protocol */
  conn->socket =
      g_socket_new (g_socket_address_get_family (saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);

  if (!conn->socket) {
    nns_loge ("Failed to create new socket");
    goto done;
  }

  if (!g_socket_connect (conn->socket, saddr, conn->cancellable, &err)) {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      nns_logd ("Cancelled connecting");
    } else {
      nns_loge ("Failed to connect to host");
    }
    goto done;
  }

  /* now connected to the requested socket */
  ret = TRUE;

done:
  g_object_unref (saddr);
  g_clear_error (&err);
  return ret;
}

/**
 * @brief Connect to the specified address.
 */
query_connection_handle
nnstreamer_query_connect (TensorQueryProtocol protocol, const char *ip,
    uint16_t port, uint32_t timeout_ms)
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
      conn->cancellable = g_cancellable_new ();
      if (!gst_tensor_query_connect (conn)) {
        nns_loge ("Failed to create new socket");
        nnstreamer_query_close (conn);
        return NULL;
      }
      break;
    }
    default:
      nns_loge ("Unsupported protocol.");
      nnstreamer_query_close (conn);
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
        nns_logd ("Failed to receive from socket");
        return -EREMOTEIO;
      }
      data->cmd = cmd;

      if (cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA) {
        /* receive size */
        if (!query_tcp_receive (conn->socket, (uint8_t *) & data->data.size,
                sizeof (data->data.size), conn->cancellable)) {
          nns_logd ("Failed to receive size from socket");
          return -EREMOTEIO;
        }
        /* receive data */
        if (!query_tcp_receive (conn->socket, (uint8_t *) data->data.data,
                data->data.size, conn->cancellable)) {
          nns_logd ("Failed to receive data from socket");
          return -EREMOTEIO;
        }
        return 0;
      } else if (data->cmd == _TENSOR_QUERY_CMD_CLIENT_ID) {
        /* receive client id */
        if (!query_tcp_receive (conn->socket, (uint8_t *) & data->client_id,
                CLIENT_ID_LEN, conn->cancellable)) {
          nns_logd ("Failed to receive client id from socket");
          return -EREMOTEIO;
        }
      } else {
        /* receive data_info */
        if (!query_tcp_receive (conn->socket, (uint8_t *) & data->data_info,
                sizeof (TensorQueryDataInfo), conn->cancellable)) {
          nns_logd ("Failed to receive data info from socket");
          return -EREMOTEIO;
        }
      }
    }
      break;
    default:
      /* NYI */
      return -EPROTONOSUPPORT;
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
        nns_logd ("Failed to send to socket");
        return -EREMOTEIO;
      }
      if (data->cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA) {
        /* send size */
        if (!query_tcp_send (conn->socket, (uint8_t *) & data->data.size,
                sizeof (data->data.size), conn->cancellable)) {
          nns_logd ("Failed to send size to socket");
          return -EREMOTEIO;
        }
        /* send data */
        if (!query_tcp_send (conn->socket, (uint8_t *) data->data.data,
                data->data.size, conn->cancellable)) {
          nns_logd ("Failed to send data to socket");
          return -EREMOTEIO;
        }
      } else if (data->cmd == _TENSOR_QUERY_CMD_CLIENT_ID) {
        /* send client id */
        if (!query_tcp_send (conn->socket, (uint8_t *) & data->client_id,
                CLIENT_ID_LEN, conn->cancellable)) {
          nns_logd ("Failed to send client id to socket");
          return -EREMOTEIO;
        }
      } else {
        /* send data_info */
        if (!query_tcp_send (conn->socket, (uint8_t *) & data->data_info,
                sizeof (TensorQueryDataInfo), conn->cancellable)) {
          nns_logd ("Failed to send data_info to socket");
          return -EREMOTEIO;
        }
      }
      break;
    default:
      /* NYI */
      return -EPROTONOSUPPORT;
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
      GError *err = NULL;
      if (!g_socket_close (conn->socket, &err)) {
        nns_loge ("Failed to close socket: %s", err->message);
        g_error_free (err);
        return -EREMOTEIO;
      }
      g_object_unref (conn->socket);
      g_object_unref (conn->cancellable);
    }
      break;
    default:
      /* NYI */
      return -EPROTONOSUPPORT;
  }
  g_free (conn->host);
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
  TensorQueryServerData *sdata = g_try_new0 (TensorQueryServerData, 1);
  if (!sdata) {
    nns_loge ("Failed to allocate server data");
    return NULL;
  }

  return (query_server_handle) sdata;
}

/**
 * @brief free server handle
 */
void
nnstreamer_query_server_data_free (query_server_handle server_data)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  if (!sdata)
    return;

  switch (sdata->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      TensorQueryConnection *conn_remained;
      while ((conn_remained = g_async_queue_try_pop (sdata->conn_queue))) {
        nnstreamer_query_close (conn_remained);
      }

      g_async_queue_unref (sdata->conn_queue);
      g_socket_listener_close (sdata->socket_listener);
      g_object_unref (sdata->socket_listener);
      g_object_unref (sdata->cancellable);
      break;
    }
    default:
      /* NYI */
      nns_loge ("Invalid protocol");
      break;
  }
  g_free (sdata);
}

/**
 * @brief set server handle params and setup server
 * @return 0 if OK, negative value if error
 */
int
nnstreamer_query_server_init (query_server_handle server_data,
    TensorQueryProtocol protocol, const char *host, uint16_t port,
    int8_t is_src)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  if (!sdata)
    return -EINVAL;
  sdata->protocol = protocol;
  sdata->is_src = is_src;

  switch (protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      GSocketAddress *saddr;
      GError *err = NULL;

      sdata->cancellable = g_cancellable_new ();
      if (!gst_tensor_query_get_saddr (host, port, sdata->cancellable, &saddr)) {
        nns_loge ("Failed to get socket address");
        return -EADDRNOTAVAIL;
      }

      sdata->socket_listener = g_socket_listener_new ();
      if (!g_socket_listener_add_address (sdata->socket_listener, saddr,
              G_SOCKET_TYPE_STREAM, G_SOCKET_PROTOCOL_TCP, NULL, NULL, &err)) {
        nns_loge ("Failed to add address: %s", err->message);
        g_clear_error (&err);
        return -EADDRNOTAVAIL;
      }
      g_socket_listener_set_backlog (sdata->socket_listener, N_BACKLOG);
      sdata->conn_queue = g_async_queue_new ();
      g_object_unref (saddr);

      g_socket_listener_accept_socket_async (sdata->socket_listener,
          sdata->cancellable, (GAsyncReadyCallback) accept_socket_async_cb,
          sdata);
    }
      break;
    default:
      /* NYI */
      nns_loge ("Invalid protocol");
      return -EPROTONOSUPPORT;
  }
  return 0;
}

/**
 * @brief accept connection from remote
 * @return query_connection_handle including connection data
 */
query_connection_handle
nnstreamer_query_server_accept (query_server_handle server_data)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  TensorQueryConnection *conn;
  if (!sdata)
    return NULL;

  switch (sdata->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      size_t size;
      GIOCondition condition;

      while (TRUE) {
        conn = g_async_queue_pop (sdata->conn_queue);

        condition = g_socket_condition_check (conn->socket,
            G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP);
        size = g_socket_get_available_bytes (conn->socket);

        if (condition && size <= 0) {
          nns_logi ("socket not available, possibly EOS");
          nnstreamer_query_close (conn);
          continue;
        }
        break;
      }
      g_async_queue_push (sdata->conn_queue, conn);
      return (query_connection_handle) conn;
    }
    default:
      /* NYI */
      nns_loge ("Invalid protocol");
      return NULL;
  }
}

/**
 * @brief [TCP] receive data for tcp server
 */
static gboolean
query_tcp_receive (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable)
{
  size_t bytes_received = 0;
  ssize_t rret;
  GError *err = NULL;
  while (bytes_received < size) {
    rret = g_socket_receive (socket, (char *) data + bytes_received,
        size - bytes_received, cancellable, &err);
    if (rret == 0) {
      nns_logi ("Connection closed");
      return FALSE;
    }
    if (rret < 0) {
      nns_loge ("Failed to read from socket: %s", err->message);
      g_clear_error (&err);
      return FALSE;
    }
    bytes_received += rret;
  }
  return TRUE;
}

/**
 * @brief [TCP] send data for tcp server
 */
static gboolean
query_tcp_send (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable)
{
  size_t bytes_sent = 0;
  ssize_t rret;
  GError *err = NULL;
  while (bytes_sent < size) {
    rret = g_socket_send (socket, (char *) data + bytes_sent,
        size - bytes_sent, cancellable, &err);
    if (rret == 0) {
      nns_logi ("Connection closed");
      return FALSE;
    }
    if (rret < 0) {
      nns_loge ("Error while sending data %s", err->message);
      g_clear_error (&err);
      return FALSE;
    }
    bytes_sent += rret;
  }
  return TRUE;
}

/**
 * @brief Generate unique id.
 */
static uint32_t
get_unique_id (GAsyncQueue * queue)
{
  TensorQueryConnection *conn;
  int32_t len = g_async_queue_length (queue), cnt = 0;
  uint32_t client_id = g_random_int ();

  while (cnt < len && (conn = g_async_queue_try_pop (queue))) {
    cnt++;
    if (conn->client_id == client_id) {
      cnt = 0;
      client_id = g_random_int ();
    }
    g_async_queue_push (queue, conn);
  }

  return client_id;
}

/**
 * @brief [TCP] Callback for socket listener that pushes socket to the queue
 */
static void
accept_socket_async_cb (GObject * source, GAsyncResult * result,
    gpointer user_data)
{
  GSocketListener *socket_listener = G_SOCKET_LISTENER (source);
  GSocket *socket;
  GSocketAddress *saddr;
  GError *err = NULL;
  TensorQueryServerData *sdata = user_data;
  TensorQueryConnection *conn;
  TensorQueryCommandData cmd_data;

  socket =
      g_socket_listener_accept_socket_finish (socket_listener, result, NULL,
      &err);
  if (!socket) {
    nns_loge ("Failed to get socket: %s", err->message);
    g_clear_error (&err);
    return;
  }

  /* create socket with connection */
  conn = g_try_new0 (TensorQueryConnection, 1);
  if (!conn) {
    nns_loge ("Failed to allocate connection");
    return;
  }
  saddr = g_socket_get_remote_address (socket, &err);
  if (!saddr) {
    nns_loge ("Failed to get socket address: %s", err->message);
    g_clear_error (&err);
    g_free (conn);
    return;
  }
  conn->protocol = (g_socket_get_protocol (socket) == G_SOCKET_PROTOCOL_TCP) ?
      _TENSOR_QUERY_PROTOCOL_TCP : _TENSOR_QUERY_PROTOCOL_END;
  conn->host = g_inet_address_to_string (g_inet_socket_address_get_address (
          (GInetSocketAddress *) saddr));
  conn->port = g_inet_socket_address_get_port ((GInetSocketAddress *) saddr);
  g_object_unref (saddr);
  conn->socket = socket;
  conn->cancellable = g_cancellable_new ();
  nns_logd ("connected from %s:%u", conn->host, conn->port);

  /** Generate and send client_id to client */
  if (sdata->is_src) {
    cmd_data.cmd = _TENSOR_QUERY_CMD_CLIENT_ID;
    cmd_data.client_id = get_unique_id (sdata->conn_queue);

    if (0 != nnstreamer_query_send (conn, &cmd_data, DEFAULT_TIMEOUT_MS)) {
      nns_loge ("Failed to send client id to client");
      nnstreamer_query_close (conn);
      return;
    }
    conn->client_id = cmd_data.client_id;
  } else {
    if (0 != nnstreamer_query_receive (conn, &cmd_data, -1)) {
      nns_loge ("Failed to receive command.");
      nnstreamer_query_close (conn);
      return;
    }
    if (cmd_data.cmd == _TENSOR_QUERY_CMD_CLIENT_ID) {
      conn->client_id = cmd_data.client_id;
      nns_logd ("Connected client id: %u", conn->client_id);
    }
  }

  g_async_queue_push (sdata->conn_queue, conn);

  g_socket_listener_accept_socket_async (socket_listener,
      sdata->cancellable, (GAsyncReadyCallback) accept_socket_async_cb, sdata);
}
