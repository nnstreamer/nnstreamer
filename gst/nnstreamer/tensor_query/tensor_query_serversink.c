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
#define DEFAULT_TIMEOUT 10

#define CAPS_STRING GST_TENSORS_CAP_DEFAULT ";" GST_TENSORS_FLEX_CAP_DEFAULT
/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

enum
{
  PROP_0,
  PROP_HOST,
  PROP_PORT,
  PROP_PROTOCOL,
  PROP_TIMEOUT
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
          3600, DEFAULT_TIMEOUT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSink", "Sink/Tensor/Query",
      "Send tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  gstbasesink_class->start = gst_tensor_query_serversink_start;
  gstbasesink_class->stop = gst_tensor_query_serversink_stop;
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
  sink->timeout = DEFAULT_TIMEOUT;
  sink->server_data = nnstreamer_query_server_data_new ();
  sink->conn_queue = g_async_queue_new ();
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_serversink_finalize (GObject * object)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (object);
  g_free (sink->host);
  nnstreamer_query_server_data_free (sink->server_data);
  sink->server_data = NULL;
  g_async_queue_unref (sink->conn_queue);
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
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief start rocessing of query_serversrc
 */
static gboolean
gst_tensor_query_serversink_start (GstBaseSink * bsink)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  GstTensorsConfig config;

  gst_tensors_config_from_peer (bsink->sinkpad, &config, NULL);
  config.rate_n = 0;
  config.rate_d = 1;
  if (!gst_tensors_config_validate (&config)) {
    nns_loge ("Invalid tensors config from peer");
    return FALSE;
  }
  gst_tensor_query_server_set_sink_config (&config);
  gst_tensors_config_free (&config);

  if (!sink->server_data) {
    nns_loge ("Server_data is NULL");
    return FALSE;
  }

  if (nnstreamer_query_server_init (sink->server_data, sink->protocol,
          sink->host, sink->port) != 0) {
    nns_loge ("Failed to setup server");
    return FALSE;
  }
  return TRUE;
}

/**
 * @brief stop processing of query_serversrc
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
 * @brief render buffer, send buffer to client
 */
static GstFlowReturn
gst_tensor_query_serversink_render (GstBaseSink * bsink, GstBuffer * buf)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  GstMetaQuery *meta_query;
  TensorQueryCommandData cmd_data;
  query_connection_handle conn;
  GstMemory *mem;
  GstMapInfo map;
  gchar *host;
  guint32 i, num_mems;

  meta_query = gst_buffer_get_meta_query (buf);
  while (TRUE) {
    conn = nnstreamer_query_server_accept (sink->server_data);
    host = nnstreamer_query_connection_get_host (conn);
    if (g_str_equal (host, meta_query->host)) {
      /* handle buffer */
      memset (&cmd_data, 0, sizeof (cmd_data));
      cmd_data.cmd = _TENSOR_QUERY_CMD_TRANSFER_START;
      num_mems = gst_buffer_n_memory (buf);
      cmd_data.data_info.num_mems = num_mems;
      for (i = 0; i < num_mems; i++) {
        mem = gst_buffer_peek_memory (buf, i);
        cmd_data.data_info.mem_sizes[i] = mem->size;
      }
      if (nnstreamer_query_send (conn, &cmd_data, sink->timeout) != 0) {
        nns_logi ("Failed to send start command");
        return GST_FLOW_EOS;
      }

      for (i = 0; i < num_mems; i++) {
        mem = gst_buffer_peek_memory (buf, i);
        if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
          nns_loge ("Failed to map memory");
          gst_memory_unref (mem);
          gst_buffer_unref (buf);
          return GST_FLOW_ERROR;
        }
        cmd_data.cmd = _TENSOR_QUERY_CMD_TRANSFER_DATA;
        cmd_data.data.size = map.size;
        cmd_data.data.data = map.data;
        if (nnstreamer_query_send (conn, &cmd_data, sink->timeout) != 0) {
          nns_logi ("Failed to send data");
          return GST_FLOW_EOS;
        }
        gst_memory_unmap (mem, &map);
      }

      cmd_data.cmd = _TENSOR_QUERY_CMD_TRANSFER_END;
      if (nnstreamer_query_send (conn, &cmd_data, sink->timeout) != 0) {
        nns_logi ("Failed to send data");
        return GST_FLOW_EOS;
      }
      g_async_queue_push (sink->conn_queue, conn);
      return GST_FLOW_OK;
    } else {
      g_async_queue_push (sink->conn_queue, conn);
    }
  }

  return GST_FLOW_ERROR;
}
