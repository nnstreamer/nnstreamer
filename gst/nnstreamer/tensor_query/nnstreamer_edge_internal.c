/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_internal.c
 * @date   6 April 2022
 * @brief  Common library to support communication among devices.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "nnstreamer_edge_common.h"
#include "nnstreamer_edge_internal.h"

#define N_BACKLOG 10
#define DEFAULT_TIMEOUT_SEC 10
#define _STR_NULL(str) ((str) ? (str) : "(NULL)")

/**
 * @brief mutex for tensor-query server table.
 */
G_LOCK_DEFINE_STATIC (_TABLE_LOCK);

/**
 * @brief Table for manage handle.
 */
static GHashTable *_table = NULL;

static void init_edge_internal (void) __attribute__((constructor));
static void fini_edge_internal (void) __attribute__((destructor));

/**
 * @brief enum for nnstreamer edge query commands.
 */
typedef enum
{
  _NNS_EDGE_CMD_TRANSFER_DATA = 0,
  _NNS_EDGE_CMD_CLIENT_ID,
  _NNS_EDGE_CMD_SRC_IP,
  _NNS_EDGE_CMD_SRC_PORT,
  _NNS_EDGE_CMD_ACCEPTABLE,
  _NNS_EDGE_CMD_END
} nns_edge_cmd_e;

/**
 * @brief Structures for tensor query command buffers.
 */
typedef struct
{
  nns_edge_cmd_e cmd;
  union
  {
    nns_edge_data_s data;
    int64_t client_id;
    int port;
  };
} nns_edge_cmd_buf_s;

/**
 * @brief Data structure for edge TCP connection.
 */
typedef struct
{
  char *ip;
  int port;
  int8_t running;
  pthread_t msg_thread;
  GSocket *socket;
  GCancellable *cancellable;
} nns_edge_conn_s;

/**
 * @brief Data structure for connection data.
 */
typedef struct
{
  nns_edge_conn_s *src_conn;
  nns_edge_conn_s *sink_conn;
  int64_t id;
} nns_edge_conn_data_s;

/**
 * @brief Structures for thread data of tensor query messge handling.
 */
typedef struct
{
  nns_edge_handle_s *eh;
  int64_t client_id;
  nns_edge_conn_s *conn;
} nns_edge_thread_data_s;

static bool _nns_edge_close_connection (nns_edge_conn_s * conn);
static int _nns_edge_send (nns_edge_conn_s * conn,
    nns_edge_cmd_buf_s * cmd_buf);
static int _nns_edge_receive (nns_edge_conn_s * conn,
    nns_edge_cmd_buf_s * cmd_buf);
static int _nns_edge_create_message_thread (nns_edge_handle_s * eh,
    nns_edge_conn_s * conn, int64_t client_id);
static int _nns_edge_tcp_connect (nns_edge_h edge_h, const char *ip, int port);

/**
 * @brief Get nnstreamer-edge connection data.
 */
static nns_edge_conn_data_s *
_nns_edge_get_conn (int64_t client_id)
{
  nns_edge_conn_data_s *data = NULL;

  G_LOCK (_TABLE_LOCK);
  data = g_hash_table_lookup (_table, GUINT_TO_POINTER (client_id));
  G_UNLOCK (_TABLE_LOCK);

  return data;
}

/**
 * @brief Get nnstreamer-edge connection data.
 */
static nns_edge_conn_data_s *
_nns_edge_add_conn (int64_t client_id)
{
  nns_edge_conn_data_s *data = NULL;

  data = _nns_edge_get_conn (client_id);

  if (NULL == data) {
    data = (nns_edge_conn_data_s *) malloc (sizeof (nns_edge_conn_data_s));
    if (NULL == data) {
      nns_edge_loge ("Failed to allocate memory for connection data.");
      return NULL;
    }

    data->id = client_id;

    G_LOCK (_TABLE_LOCK);
    g_hash_table_insert (_table, GUINT_TO_POINTER (client_id), data);
    G_UNLOCK (_TABLE_LOCK);
  }

  return data;
}

/**
 * @brief Remove nnstreamer-edge connection data.
 */
static void
_nns_edge_remove_conn (int64_t client_id)
{
  nns_edge_conn_data_s *data = _nns_edge_get_conn (client_id);
  if (NULL == data) {
    return;
  }

  G_LOCK (_TABLE_LOCK);
  g_hash_table_remove (_table, GUINT_TO_POINTER (data->id));
  G_UNLOCK (_TABLE_LOCK);

  g_free (data);
}

/**
 * @brief Disconnect all connections in edge handle.
 * @note This function should be called with handle lock.
 */
static void
_close_all_connection_in_handle (nns_edge_handle_s * eh)
{
  nns_edge_conn_data_s *conn_data;
  GList *conn_list, *l;

  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return;
  }

  /* Remove client IDs from global table. */
  G_LOCK (_TABLE_LOCK);
  /** @todo Do not remove all connections in global table later. */
  conn_list = g_hash_table_get_keys (_table);
  G_UNLOCK (_TABLE_LOCK);

  for (l = conn_list; l != NULL; l = g_list_next (l)) {
    intptr_t client_id = (intptr_t) l->data;

    conn_data = _nns_edge_get_conn (client_id);
    _nns_edge_close_connection (conn_data->sink_conn);
    _nns_edge_close_connection (conn_data->src_conn);

    _nns_edge_remove_conn (client_id);
  }

  g_list_free (conn_list);
}

/**
 * @brief Internal function to check connection.
 */
static bool
_nns_edge_check_connection (nns_edge_conn_s * conn)
{
  size_t size;
  GIOCondition condition;

  condition = g_socket_condition_check (conn->socket,
      G_IO_IN | G_IO_PRI | G_IO_ERR | G_IO_HUP);
  size = g_socket_get_available_bytes (conn->socket);

  if (condition && size <= 0) {
    nns_edge_logw ("Socket is not available, possibly EOS.");
    return false;
  }

  return true;
}

/**
 * @brief Get socket address
 */
static bool
_nns_edge_get_saddr (const char *ip, int port,
    GCancellable * cancellable, GSocketAddress ** saddr)
{
  GError *err = NULL;
  GInetAddress *addr;

  /* look up name if we need to */
  addr = g_inet_address_new_from_string (ip);
  if (!addr) {
    GList *results;
    GResolver *resolver;
    resolver = g_resolver_get_default ();
    results = g_resolver_lookup_by_name (resolver, ip, cancellable, &err);
    if (!results) {
      if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
        nns_edge_logd ("gst_tensor_query_socket_new: Cancelled name resolval");
      } else {
        nns_edge_loge ("Failed to resolve ip '%s': %s", ip, err->message);
      }
      g_clear_error (&err);
      g_object_unref (resolver);
      return false;
    }
    /** @todo Try with the second address if the first fails */
    addr = G_INET_ADDRESS (g_object_ref (results->data));
    g_resolver_free_addresses (results);
    g_object_unref (resolver);
  }

  *saddr = g_inet_socket_address_new (addr, port);
  g_object_unref (addr);

  return true;
}

/**
 * @brief Get registered handle. If not registered, create new handle and register it.
 */
int
nns_edge_create_handle (const char *id, const char *topic, nns_edge_h * edge_h)
{
  nns_edge_handle_s *eh;

  if (!id || *id == '\0') {
    nns_edge_loge ("Invalid param, given ID is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!topic || *topic == '\0') {
    nns_edge_loge ("Invalid param, given topic is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!edge_h) {
    nns_edge_loge ("Invalid param, edge_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /**
   * @todo manage edge handles
   * 1. consider adding hash table or list to manage edge handles.
   * 2. compare topic and return error if existing topic in handle is different.
   */
  eh = (nns_edge_handle_s *) malloc (sizeof (nns_edge_handle_s));
  if (!eh) {
    nns_edge_loge ("Failed to allocate memory for edge handle.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (eh, 0, sizeof (nns_edge_handle_s));
  nns_edge_lock_init (eh);
  eh->magic = NNS_EDGE_MAGIC;
  eh->id = g_strdup (id);
  eh->topic = g_strdup (topic);
  eh->protocol = NNS_EDGE_PROTOCOL_TCP;
  eh->is_server = true;
  eh->recv_ip = g_strdup ("localhost");
  eh->recv_port = 0;
  eh->caps_str = NULL;

  *edge_h = eh;
  return NNS_EDGE_ERROR_NONE;
}


/**
 * @brief [TCP] Callback for src socket listener that pushes socket to the queue
 */
static void
accept_socket_async_cb (GObject * source, GAsyncResult * result,
    gpointer user_data)
{
  GSocketListener *socket_listener = G_SOCKET_LISTENER (source);
  GSocket *socket = NULL;
  GError *err = NULL;
  nns_edge_handle_s *eh = (nns_edge_handle_s *) user_data;
  nns_edge_cmd_buf_s cmd_buf;
  nns_edge_conn_s *conn = NULL;
  bool done = false;
  gchar *connected_ip = NULL;
  int connected_port;
  nns_edge_conn_data_s *conn_data = NULL;
  int64_t client_id;

  socket =
      g_socket_listener_accept_socket_finish (socket_listener, result, NULL,
      &err);

  if (!socket) {
    nns_edge_loge ("Failed to get socket: %s", err->message);
    g_clear_error (&err);
    goto error;
  }
  g_socket_set_timeout (socket, DEFAULT_TIMEOUT_SEC);

  /* create socket with connection */
  conn = (nns_edge_conn_s *) malloc (sizeof (nns_edge_conn_s));
  if (!conn) {
    nns_edge_loge ("Failed to allocate edge connection");
    goto error;
  }
  conn->socket = socket;
  conn->cancellable = g_cancellable_new ();

  /* setting TCP_NODELAY to true in order to avoid packet batching as known as Nagle's algorithm */
  if (!g_socket_set_option (socket, IPPROTO_TCP, TCP_NODELAY, true, &err)) {
    nns_edge_loge ("Failed to set socket TCP_NODELAY option: %s", err->message);
    g_clear_error (&err);
    goto error;
  }

  /** Send caps string to check compatibility. */
  cmd_buf.cmd = _NNS_EDGE_CMD_ACCEPTABLE;
  cmd_buf.data.num = 1;
  cmd_buf.data.data[0].data = eh->caps_str;

  cmd_buf.data.data[0].data_len = strlen (cmd_buf.data.data[0].data) + 1;

  if (0 != _nns_edge_send (conn, &cmd_buf)) {
    nns_edge_loge ("Failed to send server src caps to client.");
    goto error;
  }

  cmd_buf.cmd = _NNS_EDGE_CMD_CLIENT_ID;
  cmd_buf.client_id = eh->is_server ? g_get_monotonic_time () : eh->client_id;
  client_id = cmd_buf.client_id;

  if (0 != _nns_edge_send (conn, &cmd_buf)) {
    nns_edge_loge ("Failed to send client id to client");
    goto error;
  }

  /** Receive SRC ip. */
  if (0 != _nns_edge_receive (conn, &cmd_buf) ||
      _NNS_EDGE_CMD_SRC_IP != cmd_buf.cmd) {
    nns_edge_loge ("Failed to get src IP.");
    goto error;
  }
  connected_ip = g_strdup (cmd_buf.data.data[0].data);
  g_free (cmd_buf.data.data[0].data);

  /** Receive SRC port. */
  if (0 != _nns_edge_receive (conn, &cmd_buf) ||
      _NNS_EDGE_CMD_SRC_PORT != cmd_buf.cmd) {
    nns_edge_loge ("Failed to get src port.");
    goto error;
  }
  connected_port = cmd_buf.port;

  if (0 != _nns_edge_create_message_thread (eh, conn, client_id)) {
    nns_edge_loge ("Failed to create message handle thread.");
    goto error;
  }

  done = true;
  conn_data = _nns_edge_add_conn (client_id);
  conn_data->src_conn = conn;

error:
  if (!done) {
    _nns_edge_close_connection (conn);
    return;
  }

  g_socket_listener_accept_socket_async (socket_listener,
      eh->cancellable, (GAsyncReadyCallback) accept_socket_async_cb, eh);

  if (done && eh->is_server) {
    _nns_edge_tcp_connect (eh, connected_ip, connected_port);
  }

  g_free (connected_ip);
}

/**
 * @brief Get available port number.
 */
static int
_get_available_port (void)
{
  struct sockaddr_in sin;
  int port = 0, sock;
  socklen_t len = sizeof (struct sockaddr);

  sin.sin_family = AF_INET;
  sin.sin_addr.s_addr = INADDR_ANY;
  sock = socket (AF_INET, SOCK_STREAM, 0);
  sin.sin_port = port;
  if (bind (sock, (struct sockaddr *) &sin, sizeof (struct sockaddr)) == 0) {
    getsockname (sock, (struct sockaddr *) &sin, &len);
    port = ntohs (sin.sin_port);
    nns_edge_logi ("Available port number: %d", port);
  }
  close (sock);

  return port;
}

/**
 * @brief Initialize the nnstreamer edge handle.
 */
int
nns_edge_start (nns_edge_h edge_h, bool is_server)
{
  GSocketAddress *saddr = NULL;
  GError *err = NULL;
  int ret = 0;
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  eh->is_server = is_server;
  if (!is_server && 0 == eh->recv_port)
    eh->recv_port = _get_available_port ();

  /** Initialize server src data. */
  eh->cancellable = g_cancellable_new ();
  eh->listener = g_socket_listener_new ();
  g_socket_listener_set_backlog (eh->listener, N_BACKLOG);

  if (!_nns_edge_get_saddr (eh->recv_ip, eh->recv_port, eh->cancellable,
          &saddr)) {
    nns_edge_loge ("Failed to get socket address");
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }
  if (!g_socket_listener_add_address (eh->listener, saddr,
          G_SOCKET_TYPE_STREAM, G_SOCKET_PROTOCOL_TCP, NULL, NULL, &err)) {
    nns_edge_loge ("Failed to add address: %s", err->message);
    g_clear_error (&err);
    ret = NNS_EDGE_ERROR_CONNECTION_FAILURE;
    goto error;
  }
  g_object_unref (saddr);
  saddr = NULL;

  g_socket_listener_accept_socket_async (eh->listener,
      eh->cancellable, (GAsyncReadyCallback) accept_socket_async_cb, eh);

error:
  if (saddr)
    g_object_unref (saddr);

  nns_edge_unlock (eh);
  return ret;
}

/**
 * @brief Release the given handle.
 */
int
nns_edge_release_handle (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (eh->event_cb) {
    /** @todo send new event (release handle) */
  }

  _close_all_connection_in_handle (eh);

  eh->magic = NNS_EDGE_MAGIC_DEAD;
  eh->event_cb = NULL;
  eh->user_data = NULL;
  g_free (eh->id);
  g_free (eh->topic);
  g_free (eh->recv_ip);

  nns_edge_unlock (eh);
  nns_edge_lock_destroy (eh);
  g_free (eh);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set the event callback.
 */
int
nns_edge_set_event_callback (nns_edge_h edge_h, nns_edge_event_cb cb,
    void *user_data)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (eh->event_cb) {
    /** @todo send new event (release callback) */
  }

  eh->event_cb = cb;
  eh->user_data = user_data;

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Connect to requested socket using TCP.
 */
static bool
_nns_edge_connect_socket (nns_edge_conn_s * conn)
{
  GError *err = NULL;
  GSocketAddress *saddr = NULL;
  bool ret = false;

  if (!_nns_edge_get_saddr (conn->ip, conn->port, conn->cancellable, &saddr)) {
    nns_edge_loge ("Failed to get socket address");
    return ret;
  }

  /* create sending client socket */
  conn->socket =
      g_socket_new (g_socket_address_get_family (saddr), G_SOCKET_TYPE_STREAM,
      G_SOCKET_PROTOCOL_TCP, &err);

  if (!conn->socket) {
    nns_edge_loge ("Failed to create new socket");
    goto done;
  }

  /* setting TCP_NODELAY to true in order to avoid packet batching as known as Nagle's algorithm */
  if (!g_socket_set_option (conn->socket, IPPROTO_TCP, TCP_NODELAY, true, &err)) {
    nns_edge_loge ("Failed to set socket TCP_NODELAY option: %s", err->message);
    goto done;
  }

  if (!g_socket_connect (conn->socket, saddr, conn->cancellable, &err)) {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      nns_edge_logd ("Cancelled connecting");
    } else {
      nns_edge_loge ("Failed to connect to host, %s:%d", conn->ip, conn->port);
    }
    goto done;
  }

  /* now connected to the requested socket */
  ret = true;

done:
  g_object_unref (saddr);
  g_clear_error (&err);
  return ret;
}

/**
 * @brief [TCP] send data for tcp server
 */
static bool
_send_data (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable)
{
  size_t bytes_sent = 0;
  ssize_t rret;
  GError *err = NULL;
  while (bytes_sent < size) {
    rret = g_socket_send (socket, (char *) data + bytes_sent,
        size - bytes_sent, cancellable, &err);
    if (rret == 0) {
      nns_edge_logi ("Connection closed");
      return false;
    }
    if (rret < 0) {
      nns_edge_loge ("Error while sending data %s", err->message);
      g_clear_error (&err);
      return false;
    }
    bytes_sent += rret;
  }
  return true;
}

/**
 * @brief [TCP] send data for tcp server
 */
static int
_nns_edge_send (nns_edge_conn_s * conn, nns_edge_cmd_buf_s * cmd_buf)
{
  unsigned int n;

  if (!_send_data (conn->socket, (uint8_t *) & cmd_buf->cmd,
          sizeof (nns_edge_cmd_e), conn->cancellable)) {
    nns_edge_logd ("Failed to send command to socket");
    return NNS_EDGE_ERROR_IO;
  }

  if (cmd_buf->cmd == _NNS_EDGE_CMD_TRANSFER_DATA ||
      cmd_buf->cmd == _NNS_EDGE_CMD_ACCEPTABLE ||
      cmd_buf->cmd == _NNS_EDGE_CMD_SRC_IP) {
    /** Send the number of memory. */
    if (!_send_data (conn->socket, (uint8_t *) & cmd_buf->data.num,
            sizeof (cmd_buf->data.num), conn->cancellable)) {
      nns_edge_loge ("Failed to send the number of memory to socket");
      return NNS_EDGE_ERROR_IO;
    }
    for (n = 0; n < cmd_buf->data.num; n++) {
      /* send size */
      if (!_send_data (conn->socket,
              (uint8_t *) & cmd_buf->data.data[n].data_len,
              sizeof (cmd_buf->data.data[n].data_len), conn->cancellable)) {
        nns_edge_loge ("Failed to send size to socket");
        return NNS_EDGE_ERROR_IO;
      }
      /* send data */
      if (!_send_data (conn->socket,
              (uint8_t *) cmd_buf->data.data[n].data,
              cmd_buf->data.data[n].data_len, conn->cancellable)) {
        nns_edge_loge ("Failed to send data to socket");
        return NNS_EDGE_ERROR_IO;
      }
    }
  } else if (cmd_buf->cmd == _NNS_EDGE_CMD_CLIENT_ID) {
    /* send client id */
    if (!_send_data (conn->socket, (uint8_t *) & cmd_buf->client_id,
            sizeof (cmd_buf->client_id), conn->cancellable)) {
      nns_edge_logd ("Failed to send client id to socket");
      return NNS_EDGE_ERROR_IO;
    }
  } else if (cmd_buf->cmd == _NNS_EDGE_CMD_SRC_PORT) {
    /* send client id */
    if (!_send_data (conn->socket, (uint8_t *) & cmd_buf->port,
            sizeof (cmd_buf->port), conn->cancellable)) {
      nns_edge_logd ("Failed to send client id to socket");
      return NNS_EDGE_ERROR_IO;
    }
  } else {
    nns_edge_loge ("Not supported command.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief [TCP] receive data from tcp server
 * @return 0 if OK, negative value if error
 */
static int
_receive_data (GSocket * socket, uint8_t * data, size_t size,
    GCancellable * cancellable)
{
  size_t bytes_received = 0;
  ssize_t rret;
  GError *err = NULL;

  while (bytes_received < size) {
    rret = g_socket_receive (socket, (char *) data + bytes_received,
        size - bytes_received, cancellable, &err);

    if (rret == 0) {
      nns_edge_logi ("Connection closed");
      return NNS_EDGE_ERROR_IO;
    }

    if (rret < 0) {
      nns_edge_logi ("Failed to read from socket: %s", err->message);
      g_clear_error (&err);
      return NNS_EDGE_ERROR_IO;
    }
    bytes_received += rret;
  }
  return 0;
}

/**
 * @brief receive command from connected device.
 * @return 0 if OK, negative value if error
 */
static int
_nns_edge_receive (nns_edge_conn_s * conn, nns_edge_cmd_buf_s * cmd_buf)
{
  if (_receive_data (conn->socket, (uint8_t *) & cmd_buf->cmd,
          sizeof (cmd_buf->cmd), conn->cancellable) < 0) {
    nns_edge_loge ("Failed to receive command from socket");
    return NNS_EDGE_ERROR_IO;
  }

  nns_edge_logd ("Received command: %d", cmd_buf->cmd);

  if (cmd_buf->cmd == _NNS_EDGE_CMD_TRANSFER_DATA ||
      cmd_buf->cmd == _NNS_EDGE_CMD_ACCEPTABLE ||
      cmd_buf->cmd == _NNS_EDGE_CMD_SRC_IP) {
    unsigned int n;

    /* Receive the number of memory */
    if (_receive_data (conn->socket, (uint8_t *) & cmd_buf->data.num,
            sizeof (cmd_buf->data.num), conn->cancellable) < 0) {
      nns_edge_loge ("Failed to receive data size from socket");
      return NNS_EDGE_ERROR_IO;
    }
    for (n = 0; n < cmd_buf->data.num; n++) {
      /* receive size */
      if (_receive_data (conn->socket,
              (uint8_t *) & cmd_buf->data.data[n].data_len,
              sizeof (cmd_buf->data.data[n].data_len), conn->cancellable) < 0) {
        nns_edge_loge ("Failed to receive data size from socket");
        return NNS_EDGE_ERROR_IO;
      }
      cmd_buf->data.data[n].data =
          (uint8_t *) g_malloc0 (cmd_buf->data.data[n].data_len);

      /* receive data */
      if (_receive_data (conn->socket,
              (uint8_t *) cmd_buf->data.data[n].data,
              cmd_buf->data.data[n].data_len, conn->cancellable) < 0) {
        nns_edge_loge ("Failed to receive data from socket");
        return NNS_EDGE_ERROR_IO;
      }
    }
  } else if (cmd_buf->cmd == _NNS_EDGE_CMD_CLIENT_ID) {
    /* receive client id */
    if (_receive_data (conn->socket, (uint8_t *) & cmd_buf->client_id,
            sizeof (cmd_buf->client_id), conn->cancellable) < 0) {
      nns_edge_logd ("Failed to receive client id from socket");
      return NNS_EDGE_ERROR_IO;
    }
  } else if (cmd_buf->cmd == _NNS_EDGE_CMD_SRC_PORT) {
    /* receive server src port */
    if (_receive_data (conn->socket, (uint8_t *) & cmd_buf->port,
            sizeof (cmd_buf->port), conn->cancellable) < 0) {
      nns_edge_logd ("Failed to receive sink port from socket");
      return NNS_EDGE_ERROR_IO;
    }
  } else {
    nns_edge_loge ("Not supported command.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }
  return 0;
}

/**
 * @brief [TCP] Receive buffer from the client
 * @param[in] conn connection info
 */
static void *
_message_handler (void *thread_data)
{
  nns_edge_thread_data_s *_tdata = (nns_edge_thread_data_s *) thread_data;
  nns_edge_handle_s *eh;
  nns_edge_cmd_buf_s cmd_buf;
  nns_edge_event_h event_h = NULL;

  eh = (nns_edge_handle_s *) _tdata->eh;

  if (!eh->event_cb) {
    nns_edge_loge ("NNStreamer-edge event callback is not registered.");
    goto done;
  }

  while (_tdata->conn->running) {
    guint i;
    nns_edge_data_s *data =
        (nns_edge_data_s *) malloc (sizeof (nns_edge_data_s));

    if (!data) {
      nns_edge_loge ("Failed to allocate edge data.");
      break;
    }

    /* Validate edge handle */
    if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
      nns_edge_loge ("The edge handle is invalid, it would be expired.");
      break;
    }

    if (!_nns_edge_check_connection (_tdata->conn))
      break;

    /** Receive data from the client */
    if (0 != _nns_edge_receive (_tdata->conn, &cmd_buf)) {
      nns_edge_loge ("Failed to receive data from the connected node.");
      break;
    }

    data->num = cmd_buf.data.num;
    data->client_id = _tdata->client_id;
    for (i = 0; i < data->num; i++) {
      data->data[i].data = cmd_buf.data.data[i].data;
      data->data[i].data_len = cmd_buf.data.data[i].data_len;
    }

    if (0 != nns_edge_event_create (NNS_EDGE_EVENT_NEW_DATA_RECEIVED, &event_h)) {
      nns_edge_loge ("Failed to create nns edge event.");
      goto done;
    }

    nns_edge_event_set_data (event_h, data, sizeof (data), NULL);

    if (0 != eh->event_cb (event_h, eh->user_data)) {
      nns_edge_loge ("The server is not accepted.");
      nns_edge_event_destroy (event_h);
      goto done;
    }

    nns_edge_event_destroy (event_h);
  }

done:
  _tdata->conn->running = 0;
  g_free (_tdata);
  return NULL;
}

/**
 * @brief Create message handle thread.
 */
static int
_nns_edge_create_message_thread (nns_edge_handle_s * eh, nns_edge_conn_s * conn,
    int64_t client_id)
{
  pthread_attr_t attr;
  int tid;
  nns_edge_thread_data_s *thread_data = NULL;

  thread_data =
      (nns_edge_thread_data_s *) malloc (sizeof (nns_edge_thread_data_s));
  if (!thread_data) {
    nns_edge_loge ("Failed to allocate query thread data.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

   /** Create message receving thread */
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
  conn->running = 1;
  thread_data->eh = eh;
  thread_data->conn = conn;
  thread_data->client_id = client_id;

  tid =
      pthread_create (&conn->msg_thread, &attr, _message_handler, thread_data);
  pthread_attr_destroy (&attr);

  if (tid < 0) {
    nns_edge_loge ("Failed to create message handler thread.");
    g_free (thread_data);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Connect to the destination node usig TCP.
 */
static int
_nns_edge_tcp_connect (nns_edge_h edge_h, const char *ip, int port)
{
  nns_edge_handle_s *eh;
  nns_edge_cmd_buf_s cmd_buf;
  nns_edge_conn_s *conn = NULL;
  nns_edge_event_h event_h = NULL;
  int64_t client_id;
  nns_edge_conn_data_s *conn_data;

  eh = (nns_edge_handle_s *) edge_h;

  conn = (nns_edge_conn_s *) malloc (sizeof (nns_edge_conn_s));
  if (!conn) {
    nns_edge_loge ("Failed to allocate client data.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  conn->ip = g_strdup (ip);
  conn->port = port;
  conn->cancellable = g_cancellable_new ();

  if (!_nns_edge_connect_socket (conn)) {
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  /* Get server caps string */
  if (0 != _nns_edge_receive (conn, &cmd_buf) ||
      _NNS_EDGE_CMD_ACCEPTABLE != cmd_buf.cmd) {
    nns_edge_loge ("Failed to get server src caps.");
  }

  /** Send server src and sink capability */
  if (0 != nns_edge_event_create (NNS_EDGE_EVENT_ACCEPTABLE, &event_h)) {
    nns_edge_loge ("Failed to create nns edge event.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  nns_edge_event_set_data (event_h, cmd_buf.data.data[0].data,
      cmd_buf.data.data[0].data_len, NULL);

  if (0 != eh->event_cb (event_h, eh->user_data)) {
    nns_edge_loge ("The server is not accepted.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  nns_edge_event_destroy (event_h);
  g_free (cmd_buf.data.data[0].data);

  /** Get client ID from the server */
  if (0 != _nns_edge_receive (conn, &cmd_buf) ||
      _NNS_EDGE_CMD_CLIENT_ID != cmd_buf.cmd) {
    nns_edge_loge ("Failed to get client from the server.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }
  client_id = eh->client_id = cmd_buf.client_id;

  /** Send src port.  */
  cmd_buf.cmd = _NNS_EDGE_CMD_SRC_IP;
  cmd_buf.data.data[0].data = eh->recv_ip;
  if (0 != _nns_edge_send (conn, &cmd_buf)) {
    nns_edge_loge ("Failed to send src IP address.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  /** Send src port.  */
  cmd_buf.cmd = _NNS_EDGE_CMD_SRC_PORT;
  cmd_buf.port = eh->recv_port;
  if (0 != _nns_edge_send (conn, &cmd_buf)) {
    nns_edge_loge ("Failed to send src port.");
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  conn_data = _nns_edge_add_conn (client_id);
  conn_data->sink_conn = conn;

  return 0;
}

/**
 * @brief Connect to the destination node.
 */
int
nns_edge_connect (nns_edge_h edge_h, nns_edge_protocol_e protocol,
    const char *ip, int port)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!ip || *ip == '\0') {
    nns_edge_loge ("Invalid param, given IP is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!eh->event_cb) {
    nns_edge_loge ("NNStreamer-edge event callback is not registered.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  eh->is_server = false;
  eh->protocol = protocol;

  /** Connect to info channel. */
  if (0 != _nns_edge_tcp_connect (edge_h, ip, port)) {
    nns_edge_loge ("Failed to connect to %s:%u", ip, port);
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Close connection
 */
static bool
_nns_edge_close_connection (nns_edge_conn_s * conn)
{
  GError *err = NULL;

  if (conn->socket) {
    if (!g_socket_close (conn->socket, &err)) {
      nns_edge_loge ("Failed to close src socket: %s", err->message);
      g_clear_error (&err);
      return false;
    }
    g_object_unref (conn->socket);
    conn->socket = NULL;
  }

  if (conn->cancellable) {
    g_object_unref (conn->cancellable);
    conn->cancellable = NULL;
  }

  return true;
}

/**
 * @brief Disconnect from the destination node.
 */
int
nns_edge_disconnect (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  _close_all_connection_in_handle (eh);

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Publish a message to a given topic.
 */
int
nns_edge_publish (nns_edge_h edge_h, nns_edge_data_h data_h)
{
  nns_edge_handle_s *eh;
  nns_edge_data_s *ed;

  eh = (nns_edge_handle_s *) edge_h;
  ed = (nns_edge_data_s *) data_h;

  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /** @todo update code (publish data) */

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Request result to the server.
 */
int
nns_edge_request (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data)
{
  nns_edge_handle_s *eh;
  nns_edge_data_s *ed;
  nns_edge_cmd_buf_s cmd_buf;
  nns_edge_conn_data_s *conn_data;
  unsigned int i;

  UNUSED (user_data);
  eh = (nns_edge_handle_s *) edge_h;
  ed = (nns_edge_data_s *) data_h;

  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  conn_data = _nns_edge_get_conn (eh->client_id);
  if (!_nns_edge_check_connection (conn_data->sink_conn)) {
    nns_edge_loge ("nns_edge_request: Connection failure.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_CONNECTION_FAILURE;
  }

  cmd_buf.cmd = _NNS_EDGE_CMD_TRANSFER_DATA;
  cmd_buf.data.num = ed->num;
  for (i = 0; i < ed->num; i++) {
    cmd_buf.data.data[i].data = ed->data[i].data;
    cmd_buf.data.data[i].data_len = ed->data[i].data_len;
  }
  _nns_edge_send (conn_data->sink_conn, &cmd_buf);

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Subscribe a message to a given topic.
 */
int
nns_edge_subscribe (nns_edge_h edge_h, nns_edge_data_h data_h, void *user_data)
{
  nns_edge_handle_s *eh;
  nns_edge_data_s *ed;

  UNUSED (user_data);
  eh = (nns_edge_handle_s *) edge_h;
  ed = (nns_edge_data_s *) data_h;

  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /** @todo update code (subscribe) */

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Unsubscribe a message to a given topic.
 */
int
nns_edge_unsubscribe (nns_edge_h edge_h)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /** @todo update code (unsubscribe) */

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the topic of edge handle. Caller should release returned string using free().
 * @todo is this necessary?
 */
int
nns_edge_get_topic (nns_edge_h edge_h, char **topic)
{
  nns_edge_handle_s *eh;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!topic) {
    nns_edge_loge ("Invalid param, topic should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *topic = g_strdup (eh->topic);

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set nnstreamer edge info.
 */
int
nns_edge_set_info (nns_edge_h edge_h, const char *key, const char *value)
{
  nns_edge_handle_s *eh;
  char *ret_str = NULL;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /**
   * @todo User handles (replace or append) the capability of edge handle.
   * @todo Change key-value set as json or hash table.
   */
  if (0 == g_ascii_strcasecmp (key, "CAPS")) {
    ret_str = g_strdup_printf ("%s%s", _STR_NULL (eh->caps_str), value);
    g_free (eh->caps_str);
    eh->caps_str = ret_str;
  } else if (0 == g_ascii_strcasecmp (key, "IP")) {
    g_free (eh->recv_ip);
    eh->recv_ip = g_strdup (value);
  } else if (0 == g_ascii_strcasecmp (key, "PORT")) {
    eh->recv_port = g_ascii_strtoll (value, NULL, 10);
  } else if (0 == g_ascii_strcasecmp (key, "TOPIC")) {
    g_free (eh->topic);
    eh->topic = g_strdup (value);
  } else {
    nns_edge_logw ("Failed to set edge info. Unknown key: %s", key);
  }

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Respond to a request.
 */
int
nns_edge_respond (nns_edge_h edge_h, nns_edge_data_h data_h)
{
  nns_edge_handle_s *eh;
  nns_edge_data_s *ed;
  nns_edge_conn_data_s *conn_data;
  nns_edge_cmd_buf_s cmd_buf;
  guint i;

  eh = (nns_edge_handle_s *) edge_h;
  if (!eh) {
    nns_edge_loge ("Invalid param, given edge handle is null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  nns_edge_lock (eh);

  if (!NNS_EDGE_MAGIC_IS_VALID (eh)) {
    nns_edge_loge ("Invalid param, given edge handle is invalid.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data_h) {
    nns_edge_loge ("Invalid param, data handle (data_h) should not be null.");
    nns_edge_unlock (eh);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ed = (nns_edge_data_s *) data_h;

  conn_data = _nns_edge_get_conn (ed->client_id);
  cmd_buf.cmd = _NNS_EDGE_CMD_TRANSFER_DATA;
  cmd_buf.data.num = ed->num;
  for (i = 0; i < ed->num; i++) {
    cmd_buf.data.data[i].data = ed->data[i].data;
    cmd_buf.data.data[i].data_len = ed->data[i].data_len;
  }
  _nns_edge_send (conn_data->sink_conn, &cmd_buf);

  nns_edge_unlock (eh);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Initialize the nnstreamer-edge internal.
 */
static void
init_edge_internal (void)
{
  G_LOCK (_TABLE_LOCK);
  g_assert (NULL == _table);
  _table = g_hash_table_new (g_direct_hash, g_direct_equal);
  G_UNLOCK (_TABLE_LOCK);
}

/**
 * @brief Destruct the nnstreamer-edge internal.
 */
static void
fini_edge_internal (void)
{
  G_LOCK (_TABLE_LOCK);
  g_assert (_table);
  g_hash_table_destroy (_table);
  _table = NULL;
  G_UNLOCK (_TABLE_LOCK);
}
