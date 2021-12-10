/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversink.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query server sink
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_query_serversink.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversink_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversink_debug

#define DEFAULT_HOST "localhost"
#define DEFAULT_PORT_SINK 3000
#define DEFAULT_PROTOCOL _TENSOR_QUERY_PROTOCOL_TCP
#define DEFAULT_METALESS_FRAME_LIMIT 1

/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

enum
{
  PROP_0,
  PROP_HOST,
  PROP_PORT,
  PROP_PROTOCOL,
  PROP_ID,
  PROP_TIMEOUT,
  PROP_METALESS_FRAME_LIMIT
};

#define gst_tensor_query_serversink_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSink, gst_tensor_query_serversink,
    GST_TYPE_BASE_SINK);

static void gst_tensor_query_serversink_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversink_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversink_finalize (GObject * object);

static gboolean gst_tensor_query_serversink_start (GstBaseSink * bsink);
static gboolean gst_tensor_query_serversink_stop (GstBaseSink * bsink);
static GstFlowReturn gst_tensor_query_serversink_render (GstBaseSink * bsink,
    GstBuffer * buf);
static gboolean gst_tensor_query_serversink_set_caps (GstBaseSink * basesink,
    GstCaps * caps);

/**
 * @brief initialize the class
 */
static void
gst_tensor_query_serversink_class_init (GstTensorQueryServerSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gstbasesink_class = (GstBaseSinkClass *) klass;
  gstelement_class = (GstElementClass *) gstbasesink_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_query_serversink_set_property;
  gobject_class->get_property = gst_tensor_query_serversink_get_property;
  gobject_class->finalize = gst_tensor_query_serversink_finalize;

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host", "The hostname to listen as",
          DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "The port to listen to (0=random available port)", 0,
          65535, DEFAULT_PORT_SINK,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PROTOCOL,
      g_param_spec_int ("protocol", "Protocol",
          "The network protocol to establish connection", 0,
          _TENSOR_QUERY_PROTOCOL_END, DEFAULT_PROTOCOL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TIMEOUT,
      g_param_spec_uint ("timeout", "Timeout",
          "The timeout as seconds to maintain connection", 0,
          3600, QUERY_DEFAULT_TIMEOUT_SEC,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_ID,
      g_param_spec_uint ("id", "ID",
          "ID for distinguishing query servers.", 0,
          G_MAXUINT, DEFAULT_SERVER_ID,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_METALESS_FRAME_LIMIT,
      g_param_spec_int ("limit", "Limit",
          "Limits of the number of the buffers that the server cannot handle. "
          "e.g., If the received buffer does not have a GstMetaQuery, the server cannot handle the buffer.",
          0, 65535, DEFAULT_METALESS_FRAME_LIMIT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSink", "Sink/Tensor/Query",
      "Send tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  gstbasesink_class->start = gst_tensor_query_serversink_start;
  gstbasesink_class->stop = gst_tensor_query_serversink_stop;
  gstbasesink_class->set_caps = gst_tensor_query_serversink_set_caps;
  gstbasesink_class->render = gst_tensor_query_serversink_render;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_serversink_debug,
      "tensor_query_serversink", 0, "Tensor Query Server Sink");
}

/**
 * @brief initialize the new element
 */
static void
gst_tensor_query_serversink_init (GstTensorQueryServerSink * sink)
{
  sink->host = g_strdup (DEFAULT_HOST);
  sink->port = DEFAULT_PORT_SINK;
  sink->protocol = DEFAULT_PROTOCOL;
  sink->timeout = QUERY_DEFAULT_TIMEOUT_SEC;
  sink->sink_id = DEFAULT_SERVER_ID;
  sink->server_data = nnstreamer_query_server_data_new ();
  sink->metaless_frame_count = 0;
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_serversink_finalize (GObject * object)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (object);
  g_free (sink->host);
  if (sink->server_data) {
    nnstreamer_query_server_data_free (sink->server_data);
    sink->server_data = NULL;
  }
  if (sink->server_info_h) {
    gst_tensor_query_server_remove_data (sink->server_info_h);
    sink->server_info_h = NULL;
  }
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set property
 */
static void
gst_tensor_query_serversink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSink *serversink = GST_TENSOR_QUERY_SERVERSINK (object);

  switch (prop_id) {
    case PROP_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("host property cannot be NULL");
        break;
      }
      g_free (serversink->host);
      serversink->host = g_value_dup_string (value);
      break;
    case PROP_PORT:
      serversink->port = g_value_get_uint (value);
      break;
    case PROP_PROTOCOL:
      serversink->protocol = g_value_get_int (value);
      break;
    case PROP_TIMEOUT:
      serversink->timeout = g_value_get_uint (value);
      break;
    case PROP_ID:
      serversink->sink_id = g_value_get_uint (value);
      break;
    case PROP_METALESS_FRAME_LIMIT:
      serversink->metaless_frame_limit = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property
 */
static void
gst_tensor_query_serversink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSink *serversink = GST_TENSOR_QUERY_SERVERSINK (object);

  switch (prop_id) {
    case PROP_HOST:
      g_value_set_string (value, serversink->host);
      break;
    case PROP_PORT:
      g_value_set_uint (value, serversink->port);
      break;
    case PROP_PROTOCOL:
      g_value_set_int (value, serversink->protocol);
      break;
    case PROP_TIMEOUT:
      g_value_set_uint (value, serversink->timeout);
      break;
    case PROP_ID:
      g_value_set_uint (value, serversink->sink_id);
      break;
    case PROP_METALESS_FRAME_LIMIT:
      g_value_set_int (value, serversink->metaless_frame_limit);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief start processing of query_serversink
 */
static gboolean
gst_tensor_query_serversink_start (GstBaseSink * bsink)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  GstCaps *caps;
  gchar *caps_str = NULL;

  if (!sink->server_data) {
    nns_loge ("Server_data is NULL");
    return FALSE;
  }

  if (nnstreamer_query_server_init (sink->server_data, sink->protocol,
          sink->host, sink->port, FALSE) != 0) {
    nns_loge ("Failed to setup server");
    return FALSE;
  }

  /** Set server sink information */
  sink->server_info_h = gst_tensor_query_server_add_data (sink->sink_id);
  gst_tensor_query_server_set_sink_host (sink->server_info_h, sink->host,
      sink->port);

  caps = gst_pad_get_current_caps (GST_BASE_SINK_PAD (bsink));
  if (!caps) {
    caps = gst_pad_peer_query_caps (GST_BASE_SINK_PAD (bsink), NULL);
  }

  if (caps) {
    caps_str = gst_caps_to_string (caps);
  }
  gst_tensor_query_server_set_sink_caps_str (sink->server_info_h, caps_str);

  gst_caps_unref (caps);
  g_free (caps_str);

  return TRUE;
}

/**
 * @brief stop processing of query_serversink
 */
static gboolean
gst_tensor_query_serversink_stop (GstBaseSink * bsink)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  nnstreamer_query_server_data_free (sink->server_data);
  sink->server_data = NULL;
  return TRUE;
}

/**
 * @brief An implementation of the set_caps vmethod in GstBaseSinkClass
 */
static gboolean
gst_tensor_query_serversink_set_caps (GstBaseSink * bsink, GstCaps * caps)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  gchar *caps_str;

  caps_str = gst_caps_to_string (caps);
  gst_tensor_query_server_set_sink_caps_str (sink->server_info_h, caps_str);

  g_free (caps_str);

  return TRUE;
}

/**
 * @brief render buffer, send buffer to client
 */
static GstFlowReturn
gst_tensor_query_serversink_render (GstBaseSink * bsink, GstBuffer * buf)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  GstMetaQuery *meta_query;
  query_connection_handle conn;

  meta_query = gst_buffer_get_meta_query (buf);
  if (meta_query) {
    sink->metaless_frame_count = 0;
    conn = nnstreamer_query_server_accept (sink->server_data,
        meta_query->client_id);
    if (conn) {
      if (!tensor_query_send_buffer (conn, GST_ELEMENT (sink), buf)) {
        nns_logw ("Failed to send buffer to client, drop current buffer.");
      }
    } else {
      nns_logw ("Cannot get the client connection, drop current buffer.");
    }
  } else {
    nns_logw ("Cannot get tensor query meta. Drop buffers!\n");
    sink->metaless_frame_count++;

    if (sink->metaless_frame_count >= sink->metaless_frame_limit) {
      nns_logw ("Cannot get tensor query meta. Stop the query server!\n"
          "There are elements that are not available on the query server.\n"
          "Please check available elements on the server."
          "See: https://github.com/nnstreamer/nnstreamer/wiki/Available-elements-on-query-server");
      return GST_FLOW_ERROR;
    }
  }

  return GST_FLOW_OK;
}
