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
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <nnstreamer_util.h>
#include <nnstreamer_log.h>
#include "tensor_query_common.h"
#include <pthread.h>

#define TENSOR_QUERY_SERVER_DATA_LEN 128
#define N_BACKLOG 10
#define CLIENT_ID_LEN sizeof(query_client_id_t)
#define QUERY_TIMEOUT_SEC 1

#ifndef EREMOTEIO
#define EREMOTEIO 121           /* This is Linux-specific. Define this for non-Linux systems */
#endif

/**
 * @brief Query server dependent network data
 */
typedef struct
{
  TensorQueryProtocol protocol;
  int8_t is_src;
  char *src_caps_str;
  char *sink_caps_str;
  GAsyncQueue *msg_queue;
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
  query_client_id_t client_id;
  pthread_t msg_thread;
  int8_t running;

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

/**
 * @brief Structures for tensor query message handling thread data.
 */
typedef struct
{
  TensorQueryServerData *sdata;
  TensorQueryConnection *conn;
} TensorQueryMsgThreadData;

static int
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
query_client_id_t
nnstreamer_query_connection_get_client_id (query_connection_handle connection)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;
  return conn->client_id;
}

/**
 * @brief get port from query connection handle
 */
uint16_t
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
  conn->socket =
      g_socket_new (g_socket_address_get_family (saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);
  /** @todo Support UDP protocol */

  if (!conn->socket) {
    nns_loge ("Failed to create new socket");
    goto done;
  }

  /* setting TCP_NODELAY to TRUE in order to avoid packet batching as known as Nagle's algorithm */
  if (!g_socket_set_option (conn->socket, IPPROTO_TCP, TCP_NODELAY, TRUE, &err)) {
    nns_loge ("Failed to set socket TCP_NODELAY option: %s", err->message);
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
 * @note The socket operates in two modes: blocking and non-blocking.
 *       In non-blocking mode, if there is no data available, it is immediately returned.
 */
int
nnstreamer_query_receive (query_connection_handle connection,
    TensorQueryCommandData * data)
{
  TensorQueryConnection *conn = (TensorQueryConnection *) connection;

  if (!conn) {
    nns_loge ("Invalid connection data");
    return -EINVAL;
  }
  data->protocol = conn->protocol;
  switch (conn->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      TensorQueryCommand cmd;

      if (query_tcp_receive (conn->socket, (uint8_t *) & cmd, sizeof (cmd),
              conn->cancellable) < 0) {
        nns_logd ("Failed to receive from socket");
        return -EREMOTEIO;
      }
      data->cmd = cmd;

      if (cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA ||
          cmd <= _TENSOR_QUERY_CMD_RESPOND_DENY) {
        /* receive size */
        if (query_tcp_receive (conn->socket, (uint8_t *) & data->data.size,
                sizeof (data->data.size), conn->cancellable) < 0) {
          nns_logd ("Failed to receive size from socket");
          return -EREMOTEIO;
        }

        if (cmd <= _TENSOR_QUERY_CMD_RESPOND_DENY) {
          data->data.data = (uint8_t *) malloc (data->data.size);
        }

        /* receive data */
        if (query_tcp_receive (conn->socket, (uint8_t *) data->data.data,
                data->data.size, conn->cancellable) < 0) {
          nns_loge ("Failed to receive data from socket");
          return -EREMOTEIO;
        }
        return 0;
      } else if (data->cmd == _TENSOR_QUERY_CMD_CLIENT_ID) {
        /* receive client id */
        if (query_tcp_receive (conn->socket, (uint8_t *) & data->client_id,
                CLIENT_ID_LEN, conn->cancellable) < 0) {
          nns_logd ("Failed to receive client id from socket");
          return -EREMOTEIO;
        }
      } else {
        /* receive data_info */
        if (query_tcp_receive (conn->socket, (uint8_t *) & data->data_info,
                sizeof (TensorQueryDataInfo), conn->cancellable) < 0) {
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
      if (data->cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA ||
          data->cmd <= _TENSOR_QUERY_CMD_RESPOND_DENY) {
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
  GError *err = NULL;
  int ret = 0;

  if (!conn) {
    nns_loge ("Invalid connection data");
    return -EINVAL;
  }
  switch (conn->protocol) {
    case _TENSOR_QUERY_PROTOCOL_TCP:
    {
      if (conn->socket) {
        if (!g_socket_close (conn->socket, &err)) {
          nns_loge ("Failed to close socket: %s", err->message);
          g_clear_error (&err);
        }
        g_object_unref (conn->socket);
        conn->socket = NULL;
      }

      if (conn->cancellable) {
        g_object_unref (conn->cancellable);
        conn->cancellable = NULL;
      }
      break;
    }
    default:
      /* NYI */
      ret = -EPROTONOSUPPORT;
      break;
  }

  g_free (conn->host);
  conn->host = NULL;
  g_free (conn);
  return ret;
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
      GstBuffer *buf_remained;

      if (sdata->conn_queue) {
        while ((conn_remained = g_async_queue_try_pop (sdata->conn_queue))) {
          conn_remained->running = 0;
          pthread_join (conn_remained->msg_thread, NULL);
          nnstreamer_query_close (conn_remained);
        }
        g_async_queue_unref (sdata->conn_queue);
        sdata->conn_queue = NULL;
      }

      if (sdata->is_src && sdata->msg_queue) {
        while ((buf_remained = g_async_queue_try_pop (sdata->msg_queue))) {
          gst_buffer_unref (buf_remained);
        }
        g_async_queue_unref (sdata->msg_queue);
        sdata->msg_queue = NULL;
      }

      if (sdata->socket_listener) {
        g_socket_listener_close (sdata->socket_listener);
        g_object_unref (sdata->socket_listener);
        sdata->socket_listener = NULL;
      }

      if (sdata->cancellable) {
        g_object_unref (sdata->cancellable);
        sdata->cancellable = NULL;
      }
      break;
    }
    default:
      /* NYI */
      nns_loge ("Invalid protocol");
      break;
  }
  g_free (sdata->src_caps_str);
  g_free (sdata->sink_caps_str);
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
      g_socket_listener_set_backlog (sdata->socket_listener, N_BACKLOG);

      if (!g_socket_listener_add_address (sdata->socket_listener, saddr,
              G_SOCKET_TYPE_STREAM, G_SOCKET_PROTOCOL_TCP, NULL, NULL, &err)) {
        nns_loge ("Failed to add address: %s", err->message);
        g_clear_error (&err);
        return -EADDRNOTAVAIL;
      }

      sdata->conn_queue = g_async_queue_new ();
      if (sdata->is_src)
        sdata->msg_queue = g_async_queue_new ();
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
 * @return 0 if OK, negative value if error
 */
static int
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
      return -EREMOTEIO;
    }

    if (rret < 0) {
      nns_logi ("Failed to read from socket: %s", err->message);
      g_clear_error (&err);
      return -EREMOTEIO;
    }
    bytes_received += rret;
  }
  return 0;
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
 * @brief [TCP] Receive buffer from the client
 * @param[in] conn connection info
 */
static void *
_message_handler (void *thread_data)
{
  TensorQueryMsgThreadData *_thread_data =
      (TensorQueryMsgThreadData *) thread_data;
  TensorQueryConnection *_conn = _thread_data->conn;
  TensorQueryServerData *_sdata = _thread_data->sdata;
  TensorQueryCommandData cmd_data;
  TensorQueryDataInfo data_info;
  size_t size;
  GIOCondition condition;
  GstBuffer *outbuf = NULL;
  GstMemory *mem = NULL;
  GstMapInfo map;
  GstMetaQuery *meta_query;
  uint32_t i;
  int32_t ecode;

  cmd_data.cmd = _TENSOR_QUERY_CMD_CLIENT_ID;
  cmd_data.client_id = g_get_monotonic_time ();

  if (0 != nnstreamer_query_send (_conn, &cmd_data, DEFAULT_TIMEOUT_MS)) {
    nns_loge ("Failed to send client id to client");
    goto done;
  }
  _conn->client_id = cmd_data.client_id;
  if (0 != nnstreamer_query_receive (_conn, &cmd_data)) {
    nns_logi ("Failed to receive cmd");
    goto done;
  }

  if (cmd_data.cmd == _TENSOR_QUERY_CMD_REQUEST_INFO) {
    GstCaps *sdata_caps, *cmd_caps;
    GstStructure *structure;
    gboolean result = FALSE;

    sdata_caps = gst_caps_from_string (_sdata->src_caps_str);
    cmd_caps = gst_caps_from_string ((char *) cmd_data.data.data);
    /** Server framerate may vary. Let's skip comparing the framerate. */
    gst_caps_set_simple (cmd_caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
    gst_caps_set_simple (sdata_caps, "framerate", GST_TYPE_FRACTION, 0, 1,
        NULL);

    structure = gst_caps_get_structure (sdata_caps, 0);

    free (cmd_data.data.data);

    if (gst_structure_is_tensor_stream (structure)) {
      GstTensorsConfig sdata_config, cmd_config;
      GstStructure *cmd_structure;

      gst_tensors_config_init (&cmd_config);
      gst_tensors_config_init (&sdata_config);
      cmd_structure = gst_caps_get_structure (cmd_caps, 0);

      gst_tensors_config_from_structure (&sdata_config, structure);
      gst_tensors_config_from_structure (&cmd_config, cmd_structure);

      result = gst_tensors_config_is_equal (&sdata_config, &cmd_config);
    }

    if (result || gst_caps_can_intersect (cmd_caps, sdata_caps)) {
      cmd_data.cmd = _TENSOR_QUERY_CMD_RESPOND_APPROVE;
      cmd_data.data.data = (uint8_t *) _sdata->sink_caps_str;
    } else {
      /* respond deny with src caps string */
      nns_loge ("Query caps is not acceptable!");
      nns_loge ("Query client sink caps: %s", (char *) cmd_data.data.data);
      nns_loge ("Query server src caps: %s", _sdata->src_caps_str);
      cmd_data.cmd = _TENSOR_QUERY_CMD_RESPOND_DENY;
      cmd_data.data.data = (uint8_t *) _sdata->src_caps_str;
    }
    cmd_data.data.size = (size_t) strlen ((char *) _sdata->src_caps_str);

    gst_caps_unref (sdata_caps);
    gst_caps_unref (cmd_caps);

    if (nnstreamer_query_send (_conn, &cmd_data, DEFAULT_TIMEOUT_MS) != 0) {
      nns_logi ("Failed to send respond");
      goto done;
    }
  }

  while (_conn->running) {
    condition = g_socket_condition_check (_conn->socket,
        G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP);

    size = g_socket_get_available_bytes (_conn->socket);

    if (condition && size <= 0) {
      nns_loge ("socket not available, possibly EOS");
      break;
    }

    if (0 != nnstreamer_query_receive (_conn, &cmd_data)) {
      nns_loge ("Failed to receive cmd");
      break;
    }

    if (cmd_data.cmd == _TENSOR_QUERY_CMD_TRANSFER_START) {
      data_info = cmd_data.data_info;
      outbuf = gst_buffer_new ();
      for (i = 0; i < data_info.num_mems; i++) {
        mem = gst_allocator_alloc (NULL, data_info.mem_sizes[i], NULL);
        gst_buffer_append_memory (outbuf, mem);

        if (!gst_memory_map (mem, &map, GST_MAP_READWRITE)) {
          nns_loge ("Failed to map the memory to receive data.");
          goto reset_buffer;
        }

        cmd_data.data.data = map.data;
        cmd_data.data.size = map.size;

        ecode = nnstreamer_query_receive (_conn, &cmd_data);
        gst_memory_unmap (mem, &map);

        if (ecode != 0) {
          nns_logi ("Failed to receive data");
          goto reset_buffer;
        }
      }

      /* receive end */
      if (0 != nnstreamer_query_receive (_conn, &cmd_data) ||
          cmd_data.cmd != _TENSOR_QUERY_CMD_TRANSFER_END) {
        nns_logi ("Failed to receive end command");
        goto reset_buffer;
      }

      meta_query = gst_buffer_add_meta_query (outbuf);
      if (meta_query) {
        meta_query->client_id =
            nnstreamer_query_connection_get_client_id (_conn);
      }
    } else {
      nns_logi ("Failed to receive start command.");
      break;
    }
    g_async_queue_push (_sdata->msg_queue, outbuf);
    continue;
  reset_buffer:
    gst_buffer_unref (outbuf);
  }

done:
  g_free (thread_data);
  _conn->running = 0;
  return NULL;
}

/**
 * @brief [TCP] Callback for socket listener that pushes socket to the queue
 */
static void
accept_socket_async_cb (GObject * source, GAsyncResult * result,
    gpointer user_data)
{
  GSocketListener *socket_listener = G_SOCKET_LISTENER (source);
  GSocket *socket = NULL;
  GSocketAddress *saddr = NULL;
  GError *err = NULL;
  TensorQueryServerData *sdata = user_data;
  TensorQueryConnection *conn = NULL;
  TensorQueryCommandData cmd_data;
  gboolean done = FALSE;

  socket =
      g_socket_listener_accept_socket_finish (socket_listener, result, NULL,
      &err);
  if (!socket) {
    nns_loge ("Failed to get socket: %s", err->message);
    g_clear_error (&err);
    goto error;
  }
  g_socket_set_timeout (socket, QUERY_TIMEOUT_SEC);
  /* create socket with connection */
  conn = g_try_new0 (TensorQueryConnection, 1);
  if (!conn) {
    nns_loge ("Failed to allocate connection");
    goto error;
  }

  conn->socket = socket;
  conn->cancellable = g_cancellable_new ();

  /* setting TCP_NODELAY to TRUE in order to avoid packet batching as known as Nagle's algorithm */
  if (!g_socket_set_option (socket, IPPROTO_TCP, TCP_NODELAY, TRUE, &err)) {
    nns_loge ("Failed to set socket TCP_NODELAY option: %s", err->message);
    g_clear_error (&err);
    goto error;
  }

  saddr = g_socket_get_remote_address (socket, &err);
  if (!saddr) {
    nns_loge ("Failed to get socket address: %s", err->message);
    g_clear_error (&err);
    goto error;
  }
  conn->protocol = (g_socket_get_protocol (socket) == G_SOCKET_PROTOCOL_TCP) ?
      _TENSOR_QUERY_PROTOCOL_TCP : _TENSOR_QUERY_PROTOCOL_END;
  conn->host = g_inet_address_to_string (g_inet_socket_address_get_address (
          (GInetSocketAddress *) saddr));
  conn->port = g_inet_socket_address_get_port ((GInetSocketAddress *) saddr);
  g_object_unref (saddr);
  nns_logi ("New client connected from %s:%u", conn->host, conn->port);

  /** Generate and send client_id to client */
  if (sdata->is_src) {
    TensorQueryMsgThreadData *thread_data = NULL;
    pthread_attr_t attr;
    int tid;

    thread_data = g_try_new0 (TensorQueryMsgThreadData, 1);
    if (!thread_data) {
      nns_loge ("Failed to allocate query thread data.");
      goto error;
    }
    conn->running = 1;
    thread_data->sdata = sdata;
    thread_data->conn = conn;

    pthread_attr_init (&attr);
    pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
    tid = pthread_create (&conn->msg_thread, &attr, _message_handler,
        thread_data);
    pthread_attr_destroy (&attr);

    if (tid < 0) {
      nns_loge ("Failed to create message handler thread.");
      nnstreamer_query_close (conn);
      g_free (thread_data);
      return;
    }
  } else { /** server sink */
    if (0 != nnstreamer_query_receive (conn, &cmd_data)) {
      nns_loge ("Failed to receive command.");
      goto error;
    }
    if (cmd_data.cmd == _TENSOR_QUERY_CMD_CLIENT_ID) {
      conn->client_id = cmd_data.client_id;
      nns_logd ("Connected client id: %ld", (long) conn->client_id);
    }
  }

  done = TRUE;
  g_async_queue_push (sdata->conn_queue, conn);

error:
  if (!done) {
    nnstreamer_query_close (conn);
  }

  g_socket_listener_accept_socket_async (socket_listener,
      sdata->cancellable, (GAsyncReadyCallback) accept_socket_async_cb, sdata);
}

/**
 * @brief set server source and sink caps string.
 */
void
nnstreamer_query_server_data_set_caps_str (query_server_handle server_data,
    const char *src_caps_str, const char *sink_caps_str)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;
  g_free (sdata->src_caps_str);
  g_free (sdata->sink_caps_str);
  sdata->src_caps_str = g_strdup (src_caps_str);
  sdata->sink_caps_str = g_strdup (sink_caps_str);
}

/**
 * @brief Get buffer from message queue.
 */
GstBuffer *
nnstreamer_query_server_get_buffer (query_server_handle server_data)
{
  TensorQueryServerData *sdata = (TensorQueryServerData *) server_data;

  return GST_BUFFER (g_async_queue_pop (sdata->msg_queue));
}

/**
 * @brief Send gst-buffer to destination node.
 * @return True if all data in gst-buffer is successfully sent. False if failed to transfer data.
 * @todo This function should be used in nnstreamer element. Update function name rule and params later.
 */
gboolean
tensor_query_send_buffer (query_connection_handle connection,
    GstElement * element, GstBuffer * buffer, guint timeout)
{
  TensorQueryCommandData cmd_data = { 0 };
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo map[NNS_TENSOR_SIZE_LIMIT];
  gboolean done = FALSE;
  guint i, num_mems;

  num_mems = gst_buffer_n_memory (buffer);

  /* start */
  cmd_data.cmd = _TENSOR_QUERY_CMD_TRANSFER_START;
  cmd_data.data_info.base_time = gst_element_get_base_time (element);
  cmd_data.data_info.duration = GST_BUFFER_DURATION (buffer);
  cmd_data.data_info.dts = GST_BUFFER_DTS (buffer);
  cmd_data.data_info.pts = GST_BUFFER_PTS (buffer);
  cmd_data.data_info.num_mems = num_mems;

  /* memory chunks in gst-buffer */
  for (i = 0; i < num_mems; i++) {
    mem[i] = gst_buffer_peek_memory (buffer, i);
    if (!gst_memory_map (mem[i], &map[i], GST_MAP_READ)) {
      ml_loge ("Cannot map the %uth memory in gst-buffer.", i);
      num_mems = i;
      goto error;
    }

    cmd_data.data_info.mem_sizes[i] = map[i].size;
  }

  if (nnstreamer_query_send (connection, &cmd_data, timeout) != 0) {
    nns_loge ("Failed to send start command.");
    goto error;
  }

  /* transfer data */
  cmd_data.cmd = _TENSOR_QUERY_CMD_TRANSFER_DATA;
  for (i = 0; i < num_mems; i++) {
    cmd_data.data.size = map[i].size;
    cmd_data.data.data = map[i].data;

    if (nnstreamer_query_send (connection, &cmd_data, timeout) != 0) {
      nns_loge ("Failed to send %uth data buffer", i);
      goto error;
    }
  }

  /* done */
  cmd_data.cmd = _TENSOR_QUERY_CMD_TRANSFER_END;
  if (nnstreamer_query_send (connection, &cmd_data, timeout) != 0) {
    nns_loge ("Failed to send end command.");
    goto error;
  }

  done = TRUE;

error:
  for (i = 0; i < num_mems; i++)
    gst_memory_unmap (mem[i], &map[i]);

  return done;
}
