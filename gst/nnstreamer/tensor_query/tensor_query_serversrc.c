/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversrc.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query_server src
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <nnstreamer_log.h>
#include <tensor_typedef.h>
#include "tensor_query_serversrc.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversrc_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversrc_debug

#define DEFAULT_LISTEN_HOST   NULL
#define DEFAULT_HOST          "localhost"
#define DEFAULT_PORT          4953
#define HIGHEST_PORT          65535
#define BACKLOG               1 /* client connection queue */

/**
 * @brief the capabilities of the outputs
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

/**
 * @brief query_serversrc properties
 */
enum
{
  PROP_0,
  PROP_HOST,
  PROP_PORT,
  PROP_CURRENT_PORT,
};

#define gst_tensor_query_serversrc_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSrc, gst_tensor_query_serversrc,
    GST_TYPE_PUSH_SRC);

static void gst_tensor_query_serversrc_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_finalize (GObject * object);

static gboolean gst_tensor_query_serversrc_start (GstBaseSrc * bsrc);
static gboolean gst_tensor_query_serversrc_stop (GstBaseSrc * bsrc);
static gboolean gst_tensor_query_serversrc_unlock (GstBaseSrc * bsrc);
static gboolean gst_tensor_query_serversrc_unlock_stop (GstBaseSrc * bsrc);
static GstFlowReturn gst_tensor_query_serversrc_create (GstPushSrc * psrc,
    GstBuffer ** outbuf);

/**
 * @brief initialize the query_serversrc class
 */
static void
gst_tensor_query_serversrc_class_init (GstTensorQueryServerSrcClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *gstbasesrc_class;
  GstPushSrcClass *gstpushsrc_class;

  gstpushsrc_class = (GstPushSrcClass *) klass;
  gstbasesrc_class = (GstBaseSrcClass *) gstpushsrc_class;
  gstelement_class = (GstElementClass *) gstbasesrc_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_query_serversrc_set_property;
  gobject_class->get_property = gst_tensor_query_serversrc_get_property;
  gobject_class->finalize = gst_tensor_query_serversrc_finalize;

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host", "The hostname to listen as",
          DEFAULT_LISTEN_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_int ("port", "Port",
          "The port to listen to (0=random available port)",
          0, HIGHEST_PORT, DEFAULT_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CURRENT_PORT,
      g_param_spec_int ("current-port", "current-port",
          "The port number the socket is currently bound to", 0,
          HIGHEST_PORT, 0, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSrc", "Source/Tensor/Query",
      "Receive tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  gstbasesrc_class->start = gst_tensor_query_serversrc_start;
  gstbasesrc_class->stop = gst_tensor_query_serversrc_stop;
  gstbasesrc_class->unlock = gst_tensor_query_serversrc_unlock;
  gstbasesrc_class->unlock_stop = gst_tensor_query_serversrc_unlock_stop;

  gstpushsrc_class->create = gst_tensor_query_serversrc_create;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_serversrc_debug,
      "tensor_query_serversrc", 0, "Tensor Query Server Source");
}

/**
 * @brief initialize the new query_serversrc element
 */
static void
gst_tensor_query_serversrc_init (GstTensorQueryServerSrc * self)
{
  self->server_port = DEFAULT_PORT;
  self->host = g_strdup (DEFAULT_HOST);
  self->server_socket = NULL;
  self->client_socket = NULL;
  self->cancellable = g_cancellable_new ();
}

/**
 * @brief finalize the query_serversrc object
 */
static void
gst_tensor_query_serversrc_finalize (GObject * object)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (object);

  if (src->cancellable)
    g_object_unref (src->cancellable);
  src->cancellable = NULL;
  if (src->server_socket)
    g_object_unref (src->server_socket);
  src->server_socket = NULL;
  if (src->client_socket)
    g_object_unref (src->client_socket);
  src->client_socket = NULL;
  g_free (src->host);
  src->host = NULL;

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set property of query_serversrc
 */
static void
gst_tensor_query_serversrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSrc *serversrc = GST_TENSOR_QUERY_SERVERSRC (object);

  switch (prop_id) {
    case PROP_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("host property cannot be NULL");
        break;
      }
      g_free (serversrc->host);
      serversrc->host = g_value_dup_string (value);
      break;
    case PROP_PORT:
      serversrc->server_port = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property of query_serversrc
 */
static void
gst_tensor_query_serversrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSrc *serversrc = GST_TENSOR_QUERY_SERVERSRC (object);

  switch (prop_id) {
    case PROP_HOST:
      g_value_set_string (value, serversrc->host);
      break;
    case PROP_PORT:
      g_value_set_int (value, serversrc->server_port);
      break;
    case PROP_CURRENT_PORT:
      g_value_set_int (value, g_atomic_int_get (&serversrc->current_port));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief start processing of query_serversrc, setting up the server
 */
static gboolean
gst_tensor_query_serversrc_start (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);
  GError *err = NULL;
  GSocketAddress *saddr = NULL;
  gint bound_port = 0;

  src->bytes_received = 0;

  src->server_socket =
      gst_tensor_query_socket_new (src->host, src->server_port,
      src->cancellable, &saddr);

  if (!src->server_socket) {
    nns_loge ("Failed to create socket: %s", err->message);
    g_clear_error (&err);
    return FALSE;
  }

  nns_logd ("opened receiving server socket");

  /* bind */
  nns_logd ("binding server socket to address");
  if (!g_socket_bind (src->server_socket, saddr, TRUE, &err)) {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      nns_logw ("Cancelled binding");
    } else {
      nns_loge ("Failed to bind on host '%s:%d': %s", src->host,
          src->server_port, err->message);
    }
    g_clear_error (&err);
    g_object_unref (saddr);
    gst_tensor_query_serversrc_stop (GST_BASE_SRC (src));
    return FALSE;
  }

  g_object_unref (saddr);
  nns_logd ("listening on server socket");

  g_socket_set_listen_backlog (src->server_socket, BACKLOG);

  if (!g_socket_listen (src->server_socket, &err)) {
    if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
      nns_logd ("Cancelled listening");
    } else {
      nns_loge ("Failed to listen on host '%s:%d': %s", src->host,
          src->server_port, err->message);
    }
    g_clear_error (&err);
    gst_tensor_query_serversrc_stop (GST_BASE_SRC (src));
    return FALSE;
  }

  if (src->server_port == 0) {
    saddr = g_socket_get_local_address (src->server_socket, NULL);
    bound_port = g_inet_socket_address_get_port ((GInetSocketAddress *) saddr);
    g_object_unref (saddr);
  } else {
    bound_port = src->server_port;
  }

  nns_logd ("listening on port %d", bound_port);

  g_atomic_int_set (&src->current_port, bound_port);
  g_object_notify (G_OBJECT (src), "current-port");

  return TRUE;
}

/**
 * @brief stop processing of query_serversrc
 */
static gboolean
gst_tensor_query_serversrc_stop (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);
  GError *err = NULL;

  if (src->client_socket) {
    nns_logd ("closing socket");

    if (!g_socket_close (src->client_socket, &err)) {
      nns_loge ("Failed to close socket: %s", err->message);
      g_clear_error (&err);
    }
    g_object_unref (src->client_socket);
    src->client_socket = NULL;
  }

  if (src->server_socket) {
    nns_logd ("closing socket");

    if (!g_socket_close (src->server_socket, &err)) {
      nns_loge ("Failed to close socket: %s", err->message);
      g_clear_error (&err);
    }
    g_object_unref (src->server_socket);
    src->server_socket = NULL;

    g_atomic_int_set (&src->current_port, 0);
    g_object_notify (G_OBJECT (src), "current-port");
  }

  return TRUE;
}

/**
 * @brief unlock pending access of query_serversrc
 */
static gboolean
gst_tensor_query_serversrc_unlock (GstBaseSrc * bsrc)
{
  /* called only between start() and stop() */
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);
  g_cancellable_cancel (src->cancellable);
  return TRUE;
}

/**
 * @brief clear the unlock request of query_serversrc
 */
static gboolean
gst_tensor_query_serversrc_unlock_stop (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);

  g_object_unref (src->cancellable);
  src->cancellable = g_cancellable_new ();

  return TRUE;
}

/**
 * @brief create query_serversrc, wait on socket and receive data
 */
static GstFlowReturn
gst_tensor_query_serversrc_create (GstPushSrc * psrc, GstBuffer ** outbuf)
{
  GstTensorQueryServerSrc *src;
  GstFlowReturn ret = GST_FLOW_OK;
  GError *err = NULL;

  src = GST_TENSOR_QUERY_SERVERSRC (psrc);

  if (!src->client_socket) {
    /* wait for connections on server socket */
    src->client_socket =
        g_socket_accept (src->server_socket, src->cancellable, &err);

    if (!src->client_socket) {
      if (g_error_matches (err, G_IO_ERROR, G_IO_ERROR_CANCELLED)) {
        nns_logw ("Cancelled accepting of client");
        ret = GST_FLOW_FLUSHING;
      } else {
        nns_loge ("Failed to accept client: %s", err->message);
        ret = GST_FLOW_ERROR;
      }
      g_clear_error (&err);
      return ret;
    }
    nns_logd ("closing server socket");

    if (!g_socket_close (src->server_socket, &err)) {
      nns_loge ("Failed to close socket: %s", err->message);
      g_clear_error (&err);
    }
  }

  /* read from the socket */
  nns_logd ("asked for a buffer");
  *outbuf = gst_buffer_new_and_alloc (0UL);
  ret = gst_tensor_query_socket_receive (src->client_socket, src->cancellable,
      &src->bytes_received, *outbuf);

  return ret;
}
