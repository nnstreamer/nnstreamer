/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_client.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query client
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "nnstreamer_util.h"
#include "tensor_query_client.h"
#include <gio/gio.h>
#include <glib.h>
#include <string.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Properties.
 */
enum
{
  PROP_0,
  PROP_SINK_HOST,
  PROP_SINK_PORT,
  PROP_SRC_HOST,
  PROP_SRC_PORT,
  PROP_PROTOCOL,
  PROP_OPERATION,
  PROP_BROKER_HOST,
  PROP_BROKER_PORT,
  PROP_SILENT,
};

#define TCP_HIGHEST_PORT        65535
#define TCP_DEFAULT_HOST        "localhost"
#define TCP_DEFAULT_SINK_PORT        3000
#define TCP_DEFAULT_SRC_PORT        3001
#define DEFAULT_SILENT TRUE

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_client_debug);
#define GST_CAT_DEFAULT gst_tensor_query_client_debug

/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

/**
 * @brief the capabilities of the outputs.
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

#define gst_tensor_query_client_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryClient, gst_tensor_query_client, GST_TYPE_ELEMENT);

static void gst_tensor_query_client_finalize (GObject * object);
static void gst_tensor_query_client_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_client_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

static gboolean gst_tensor_query_client_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_query_client_sink_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static GstFlowReturn gst_tensor_query_client_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buf);
static GstCaps *gst_tensor_query_client_query_caps (GstTensorQueryClient * self,
    GstPad * pad, GstCaps * filter);

/**
 * @brief initialize the class
 */
static void
gst_tensor_query_client_class_init (GstTensorQueryClientClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_tensor_query_client_set_property;
  gobject_class->get_property = gst_tensor_query_client_get_property;
  gobject_class->finalize = gst_tensor_query_client_finalize;

  /** install property goes here */
  g_object_class_install_property (gobject_class, PROP_SINK_HOST,
      g_param_spec_string ("sink-host", "Sink Host",
          "A tenor query sink host to send the packets to/from",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SINK_PORT,
      g_param_spec_uint ("sink-port", "Sink Port",
          "The port of tensor query sink to send the packets to/from", 0,
          TCP_HIGHEST_PORT, TCP_DEFAULT_SINK_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SRC_HOST,
      g_param_spec_string ("src-host", "Source Host",
          "A tenor query src host to send the packets to/from",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SRC_PORT,
      g_param_spec_uint ("src-port", "Source Port",
          "The port of tensor query src to send the packets to/from", 0,
          TCP_HIGHEST_PORT, TCP_DEFAULT_SRC_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PROTOCOL,
      g_param_spec_enum ("protocol", "Protocol",
          "The network protocol to establish connections between client and server.",
          GST_TYPE_QUERY_PROTOCOL, DEFAULT_PROTOCOL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OPERATION,
      g_param_spec_string ("operation", "Operation",
          "The main operation of the host.",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_BROKER_HOST,
      g_param_spec_string ("broker-host", "Broker Host",
          "Broker host address to connect.", DEFAULT_BROKER_HOST,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_BROKER_PORT,
      g_param_spec_uint ("broker-port", "Broker Port",
          "Broker port to connect.", 0, 65535,
          DEFAULT_BROKER_PORT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryClient", "Filter/Tensor/Query",
      "Handle querying tensor data through the network",
      "Samsung Electronics Co., Ltd.");

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_client_debug, "tensor_query_client",
      0, "Tensor Query Client");
}

/**
 * @brief initialize the new element
 */
static void
gst_tensor_query_client_init (GstTensorQueryClient * self)
{
  /** setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sinktemplate, "sink");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_query_client_sink_event));
  gst_pad_set_query_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_query_client_sink_query));
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_query_client_chain));

  /** setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&srctemplate, "src");
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /* init properties */
  self->silent = DEFAULT_SILENT;
  self->protocol = DEFAULT_PROTOCOL;
  self->sink_conn = NULL;
  self->sink_host = g_strdup (TCP_DEFAULT_HOST);
  self->sink_port = TCP_DEFAULT_SINK_PORT;
  self->src_conn = NULL;
  self->src_host = g_strdup (TCP_DEFAULT_HOST);
  self->src_port = TCP_DEFAULT_SRC_PORT;
  self->operation = NULL;
  self->broker_host = g_strdup (DEFAULT_BROKER_HOST);
  self->broker_port = DEFAULT_BROKER_PORT;
  self->in_caps_str = NULL;

  tensor_query_hybrid_init (&self->hybrid_info, NULL, 0, FALSE);
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_client_finalize (GObject * object)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (object);

  g_free (self->sink_host);
  self->sink_host = NULL;
  g_free (self->src_host);
  self->src_host = NULL;
  g_free (self->operation);
  self->operation = NULL;
  g_free (self->broker_host);
  self->broker_host = NULL;
  g_free (self->in_caps_str);
  self->in_caps_str = NULL;
  tensor_query_hybrid_close (&self->hybrid_info);
  if (self->sink_conn) {
    nnstreamer_query_close (self->sink_conn);
    self->sink_conn = NULL;
  }
  if (self->src_conn) {
    nnstreamer_query_close (self->src_conn);
    self->src_conn = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set property
 */
static void
gst_tensor_query_client_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (object);

  switch (prop_id) {
    case PROP_SINK_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("Sink host property cannot be NULL");
        break;
      }
      g_free (self->sink_host);
      self->sink_host = g_value_dup_string (value);
      break;
    case PROP_SINK_PORT:
      self->sink_port = g_value_get_uint (value);
      break;
    case PROP_SRC_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("Source host property cannot be NULL");
        break;
      }
      g_free (self->src_host);
      self->src_host = g_value_dup_string (value);
      break;
    case PROP_SRC_PORT:
      self->src_port = g_value_get_uint (value);
      break;
    case PROP_PROTOCOL:
    {
        /** @todo expand when other protocols are ready */
      TensorQueryProtocol protocol = g_value_get_enum (value);
      if (protocol == _TENSOR_QUERY_PROTOCOL_TCP)
        self->protocol = protocol;
    }
      break;
    case PROP_OPERATION:
      if (!g_value_get_string (value)) {
        nns_logw
            ("Operation property cannot be NULL. Query-hybrid is disabled.");
        break;
      }
      g_free (self->operation);
      self->operation = g_value_dup_string (value);
      break;
    case PROP_BROKER_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("Broker host property cannot be NULL");
        break;
      }
      g_free (self->broker_host);
      self->broker_host = g_value_dup_string (value);
      break;
    case PROP_BROKER_PORT:
      self->broker_port = g_value_get_uint (value);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
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
gst_tensor_query_client_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (object);

  switch (prop_id) {
    case PROP_SINK_HOST:
      g_value_set_string (value, self->sink_host);
      break;
    case PROP_SINK_PORT:
      g_value_set_uint (value, self->sink_port);
      break;
    case PROP_SRC_HOST:
      g_value_set_string (value, self->src_host);
      break;
    case PROP_SRC_PORT:
      g_value_set_uint (value, self->src_port);
      break;
    case PROP_PROTOCOL:
      g_value_set_enum (value, self->protocol);
      break;
    case PROP_OPERATION:
      g_value_set_string (value, self->operation);
      break;
    case PROP_BROKER_HOST:
      g_value_set_string (value, self->broker_host);
      break;
    case PROP_BROKER_PORT:
      g_value_set_uint (value, self->broker_port);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Update src pad caps from tensors config.
 */
static gboolean
gst_tensor_query_client_update_caps (GstTensorQueryClient * self,
    const gchar * caps_str)
{
  GstCaps *curr_caps, *out_caps;
  gboolean ret = FALSE;
  out_caps = gst_caps_from_string (caps_str);
  silent_debug_caps (self, out_caps, "set out-caps");

  /* Update src pad caps if it is different. */
  curr_caps = gst_pad_get_current_caps (self->srcpad);
  if (curr_caps == NULL || !gst_caps_is_equal (curr_caps, out_caps)) {
    if (gst_caps_is_fixed (out_caps)) {
      ret = gst_pad_set_caps (self->srcpad, out_caps);
    } else {
      nns_loge ("out-caps from tensor_query_serversink is not fixed. "
          "Failed to update client src caps, out-caps: %s", caps_str);
    }
  } else {
    /** Don't need to update when the capability is the same. */
    ret = TRUE;
  }

  if (curr_caps)
    gst_caps_unref (curr_caps);

  gst_caps_unref (out_caps);

  return ret;
}

/**
 * @brief Connect to query server. (Direct connection)
 */
static gboolean
_connect_to_server (GstTensorQueryClient * self)
{
  TensorQueryCommandData cmd_buf;
  query_client_id_t client_id;

  nns_logd ("Server src info: %s:%u", self->src_host, self->src_port);
  self->src_conn = nnstreamer_query_connect (self->protocol, self->src_host,
      self->src_port, QUERY_DEFAULT_TIMEOUT_SEC);
  if (!self->src_conn) {
    nns_loge ("Failed to connect server source ");
    return FALSE;
  }

  /** Receive client ID from server src */
  if (0 != nnstreamer_query_receive (self->src_conn, &cmd_buf) ||
      cmd_buf.cmd != _TENSOR_QUERY_CMD_CLIENT_ID) {
    nns_loge ("Failed to receive client ID.");
    return FALSE;
  }

  client_id = cmd_buf.client_id;
  nnstreamer_query_set_client_id (self->src_conn, client_id);

  cmd_buf.cmd = _TENSOR_QUERY_CMD_REQUEST_INFO;
  cmd_buf.data.data = (uint8_t *) self->in_caps_str;
  cmd_buf.data.size = (size_t) strlen (self->in_caps_str) + 1;

  if (0 != nnstreamer_query_send (self->src_conn, &cmd_buf)) {
    nns_loge ("Failed to send request info cmd buf");
    return FALSE;
  }

  if (0 != nnstreamer_query_receive (self->src_conn, &cmd_buf)) {
    nns_loge ("Failed to receive response from the query server.");
    return FALSE;
  }

  if (cmd_buf.cmd == _TENSOR_QUERY_CMD_RESPOND_APPROVE) {
    if (!gst_tensor_query_client_update_caps (self, (char *) cmd_buf.data.data)) {
      nns_loge ("Failed to update client source caps.");
      return FALSE;
    }
  } else {
    /** @todo Retry for info */
    nns_loge ("Failed to receive approve command.");
    return FALSE;
  }

  nns_logd ("Server sink info: %s:%u", self->sink_host, self->sink_port);
  self->sink_conn =
      nnstreamer_query_connect (self->protocol, self->sink_host,
      self->sink_port, QUERY_DEFAULT_TIMEOUT_SEC);
  if (!self->sink_conn) {
    nns_loge ("Failed to connect server sink ");
    return FALSE;
  }

  nnstreamer_query_set_client_id (self->sink_conn, client_id);
  cmd_buf.cmd = _TENSOR_QUERY_CMD_CLIENT_ID;
  cmd_buf.client_id = client_id;
  if (0 != nnstreamer_query_send (self->sink_conn, &cmd_buf)) {
    nns_loge ("Failed to send client ID to server sink");
    return FALSE;
  }
  return TRUE;
}

/**
 * @brief Copy server info.
 */
static void
_copy_srv_info (GstTensorQueryClient * self, query_server_info_s * server)
{
  g_free (self->src_host);
  self->src_host = g_strdup (server->src.host);
  self->src_port = server->src.port;
  g_free (self->sink_host);
  self->sink_host = g_strdup (server->sink.host);
  self->sink_port = server->sink.port;
}

/**
 * @brief Retry to connect to available server.
 */
static gboolean
_client_retry_connection (GstTensorQueryClient * self)
{
  gboolean ret = FALSE;
  query_server_info_s *server = NULL;

  g_return_val_if_fail (self->operation, FALSE);
  nns_logd ("Retry to connect to available server.");

  while ((server = tensor_query_hybrid_get_server_info (&self->hybrid_info))) {
    nnstreamer_query_close (self->sink_conn);
    nnstreamer_query_close (self->src_conn);
    self->sink_conn = NULL;
    self->src_conn = NULL;

    _copy_srv_info (self, server);
    tensor_query_hybrid_free_server_info (server);

    if (_connect_to_server (self)) {
      nns_logi ("Connected to new server. src: %s:%u, sink: %s:%u",
          self->src_host, self->src_port, self->sink_host, self->sink_port);
      ret = TRUE;
      break;
    }
  }

  return ret;
}

/**
 * @brief This function handles sink event.
 */
static gboolean
gst_tensor_query_client_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (parent);
  gboolean ret = TRUE;

  GST_DEBUG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);

      /** Subscribe server info from broker */
      if (self->operation) {
        query_server_info_s *server;

        tensor_query_hybrid_set_broker (&self->hybrid_info,
            self->broker_host, self->broker_port);

        if (!tensor_query_hybrid_subscribe (&self->hybrid_info,
                self->operation)) {
          nns_loge ("Failed to subscribe a topic.");
          gst_event_unref (event);
          return FALSE;
        }

        server = tensor_query_hybrid_get_server_info (&self->hybrid_info);
        if (server) {
          _copy_srv_info (self, server);
          tensor_query_hybrid_free_server_info (server);
        }
      } else {
        nns_logi ("Query-hybrid feature is disabled.");
        nns_logi
            ("Specify operation to subscribe to the available broker (e.g., operation=object_detection).");
      }

      g_free (self->in_caps_str);
      self->in_caps_str = gst_caps_to_string (caps);

      if (!_connect_to_server (self)) {
        ret = _client_retry_connection (self);
      }

      gst_event_unref (event);
      return ret;
    }
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief This function handles sink pad query.
 */
static gboolean
gst_tensor_query_client_sink_query (GstPad * pad,
    GstObject * parent, GstQuery * query)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (parent);

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      caps = gst_tensor_query_client_query_caps (self, pad, filter);

      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      return TRUE;
    }
    case GST_QUERY_ACCEPT_CAPS:
    {
      GstCaps *caps;
      GstCaps *template_caps;
      gboolean res = FALSE;

      gst_query_parse_accept_caps (query, &caps);
      silent_debug_caps (self, caps, "accept-caps");

      if (gst_caps_is_fixed (caps)) {
        template_caps = gst_pad_get_pad_template_caps (pad);

        res = gst_caps_can_intersect (template_caps, caps);
        gst_caps_unref (template_caps);
      }

      gst_query_set_accept_caps_result (query, res);
      return TRUE;
    }
    default:
      break;
  }

  return gst_pad_query_default (pad, parent, query);
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_tensor_query_client_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buf)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (parent);
  GstBuffer *out_buf = NULL;
  GstFlowReturn res = GST_FLOW_OK;

  UNUSED (pad);

  if (!tensor_query_send_buffer (self->src_conn, GST_ELEMENT (self), buf)) {
    nns_logw ("Failed to send buffer to server node, retry connection.");
    goto retry;
  }

  out_buf = tensor_query_receive_buffer (self->sink_conn);
  if (out_buf) {
    /* metadata from incoming buffer */
    gst_buffer_copy_into (out_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

    res = gst_pad_push (self->srcpad, out_buf);
    goto done;
  }

  nns_logw ("Failed to receive result from server node, retry connection.");

retry:
  if (!self->operation || !_client_retry_connection (self)) {
    nns_loge ("Failed to retry connection");
    res = GST_FLOW_ERROR;
  }
done:
  gst_buffer_unref (buf);
  return res;
}

/**
 * @brief Get pad caps for caps negotiation.
 */
static GstCaps *
gst_tensor_query_client_query_caps (GstTensorQueryClient * self, GstPad * pad,
    GstCaps * filter)
{
  GstCaps *caps;

  caps = gst_pad_get_current_caps (pad);
  if (!caps) {
    /** pad don't have current caps. use the template caps */
    caps = gst_pad_get_pad_template_caps (pad);
  }

  silent_debug_caps (self, caps, "caps");
  silent_debug_caps (self, filter, "filter");

  if (filter) {
    GstCaps *intersection;
    intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }

  silent_debug_caps (self, caps, "result");
  return caps;
}
