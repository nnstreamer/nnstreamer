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
#include "tensor_query_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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
  PROP_HOST,
  PROP_PORT,
  PROP_DEST_HOST,
  PROP_DEST_PORT,
  PROP_CONNECT_TYPE,
  PROP_TOPIC,
  PROP_TIMEOUT,
  PROP_SILENT,
};

#define TCP_HIGHEST_PORT        65535
#define TCP_DEFAULT_HOST        "localhost"
#define TCP_DEFAULT_SRV_SRC_PORT 3000
#define TCP_DEFAULT_CLIENT_SRC_PORT 3001
#define DEFAULT_CLIENT_TIMEOUT  0
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
  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host",
          "A host address to receive the packets from query server",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "A port number to receive the packets from query server", 0,
          TCP_HIGHEST_PORT, TCP_DEFAULT_SRV_SRC_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_DEST_HOST,
      g_param_spec_string ("dest-host", "Destination Host",
          "A tenor query server host to send the packets",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_DEST_PORT,
      g_param_spec_uint ("dest-port", "Destination Port",
          "The port of tensor query server to send the packets", 0,
          TCP_HIGHEST_PORT, TCP_DEFAULT_CLIENT_SRC_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CONNECT_TYPE,
      g_param_spec_enum ("connect-type", "Connect Type",
          "The connections type between client and server.",
          GST_TYPE_QUERY_CONNECT_TYPE, DEFAULT_CONNECT_TYPE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TOPIC,
      g_param_spec_string ("topic", "Topic",
          "The main topic of the host.",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TIMEOUT,
      g_param_spec_uint ("timeout", "timeout value",
          "A timeout value (in ms) to wait message from query server after sending buffer to server. 0 means no wait.",
          0, G_MAXUINT, DEFAULT_CLIENT_TIMEOUT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  self->connect_type = DEFAULT_CONNECT_TYPE;
  self->host = g_strdup (TCP_DEFAULT_HOST);
  self->port = TCP_DEFAULT_CLIENT_SRC_PORT;
  self->dest_host = g_strdup (TCP_DEFAULT_HOST);
  self->dest_port = TCP_DEFAULT_SRV_SRC_PORT;
  self->topic = NULL;
  self->in_caps_str = NULL;
  self->timeout = DEFAULT_CLIENT_TIMEOUT;
  self->edge_h = NULL;
  self->msg_queue = g_async_queue_new ();
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_client_finalize (GObject * object)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (object);
  nns_edge_data_h data_h;

  g_free (self->host);
  self->host = NULL;
  g_free (self->dest_host);
  self->dest_host = NULL;
  g_free (self->topic);
  self->topic = NULL;
  g_free (self->in_caps_str);
  self->in_caps_str = NULL;

  while ((data_h = g_async_queue_try_pop (self->msg_queue))) {
    nns_edge_data_destroy (data_h);
  }

  if (self->msg_queue) {
    g_async_queue_unref (self->msg_queue);
    self->msg_queue = NULL;
  }

  if (self->edge_h) {
    nns_edge_release_handle (self->edge_h);
    self->edge_h = NULL;
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

  /** @todo DO NOT update properties (host, port, ..) while pipeline is running. */
  switch (prop_id) {
    case PROP_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("Sink host property cannot be NULL");
        break;
      }
      g_free (self->host);
      self->host = g_value_dup_string (value);
      break;
    case PROP_PORT:
      self->port = g_value_get_uint (value);
      break;
    case PROP_DEST_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("Sink host property cannot be NULL");
        break;
      }
      g_free (self->dest_host);
      self->dest_host = g_value_dup_string (value);
      break;
    case PROP_DEST_PORT:
      self->dest_port = g_value_get_uint (value);
      break;
    case PROP_CONNECT_TYPE:
      self->connect_type = g_value_get_enum (value);
      break;
    case PROP_TOPIC:
      if (!g_value_get_string (value)) {
        nns_logw ("Topic property cannot be NULL. Query-hybrid is disabled.");
        break;
      }
      g_free (self->topic);
      self->topic = g_value_dup_string (value);
      break;
    case PROP_TIMEOUT:
      self->timeout = g_value_get_uint (value);
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
    case PROP_HOST:
      g_value_set_string (value, self->host);
      break;
    case PROP_PORT:
      g_value_set_uint (value, self->port);
      break;
    case PROP_DEST_HOST:
      g_value_set_string (value, self->dest_host);
      break;
    case PROP_DEST_PORT:
      g_value_set_uint (value, self->dest_port);
      break;
    case PROP_CONNECT_TYPE:
      g_value_set_enum (value, self->connect_type);
      break;
    case PROP_TOPIC:
      g_value_set_string (value, self->topic);
      break;
    case PROP_TIMEOUT:
      g_value_set_uint (value, self->timeout);
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
 * @brief Retry to connect to available server.
 */
static gboolean
_client_retry_connection (GstTensorQueryClient * self)
{
  if (NNS_EDGE_ERROR_NONE != nns_edge_disconnect (self->edge_h)) {
    nns_loge ("Failed to retry connection, disconnection failure");
    return FALSE;
  }

  if (NNS_EDGE_ERROR_NONE != nns_edge_connect (self->edge_h,
          self->dest_host, self->dest_port)) {
    nns_loge ("Failed to retry connection, connection failure");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Parse caps from received event data.
 */
static gchar *
_nns_edge_parse_caps (gchar * caps_str, gboolean is_src)
{
  gchar **strv;
  gint num, i;
  gchar *find_key = NULL;
  gchar *ret_str = NULL;

  if (!caps_str)
    return NULL;

  strv = g_strsplit (caps_str, "@", -1);
  num = g_strv_length (strv);

  find_key =
      is_src ==
      TRUE ? g_strdup ("query_server_src_caps") :
      g_strdup ("query_server_sink_caps");

  for (i = 1; i < num; i += 2) {
    if (0 == g_strcmp0 (find_key, strv[i])) {
      ret_str = g_strdup (strv[i + 1]);
      break;
    }
  }

  g_free (find_key);
  g_strfreev (strv);

  return ret_str;
}

/**
 * @brief nnstreamer-edge event callback.
 */
static int
_nns_edge_event_cb (nns_edge_event_h event_h, void *user_data)
{
  nns_edge_event_e event_type;
  int ret = NNS_EDGE_ERROR_NONE;
  GstTensorQueryClient *self = (GstTensorQueryClient *) user_data;

  if (NNS_EDGE_ERROR_NONE != nns_edge_event_get_type (event_h, &event_type)) {
    nns_loge ("Failed to get event type!");
    return NNS_EDGE_ERROR_NOT_SUPPORTED;
  }

  switch (event_type) {
    case NNS_EDGE_EVENT_CAPABILITY:
    {
      GstCaps *server_caps, *client_caps;
      GstStructure *server_st, *client_st;
      gboolean result = FALSE;
      gchar *ret_str, *caps_str;

      nns_edge_event_parse_capability (event_h, &caps_str);
      ret_str = _nns_edge_parse_caps (caps_str, TRUE);
      nns_logd ("Received server-src caps: %s", GST_STR_NULL (ret_str));
      client_caps = gst_caps_from_string ((gchar *) self->in_caps_str);
      server_caps = gst_caps_from_string (ret_str);
      g_free (ret_str);

      /** Server framerate may vary. Let's skip comparing the framerate. */
      gst_caps_set_simple (server_caps, "framerate", GST_TYPE_FRACTION, 0, 1,
          NULL);
      gst_caps_set_simple (client_caps, "framerate", GST_TYPE_FRACTION, 0, 1,
          NULL);

      server_st = gst_caps_get_structure (server_caps, 0);
      client_st = gst_caps_get_structure (client_caps, 0);

      if (gst_structure_is_tensor_stream (server_st)) {
        GstTensorsConfig server_config, client_config;

        gst_tensors_config_from_structure (&server_config, server_st);
        gst_tensors_config_from_structure (&client_config, client_st);

        result = gst_tensors_config_is_equal (&server_config, &client_config);
      }

      if (result || gst_caps_can_intersect (client_caps, server_caps)) {
        /** Update client src caps */
        ret_str = _nns_edge_parse_caps (caps_str, FALSE);
        nns_logd ("Received server-sink caps: %s", GST_STR_NULL (ret_str));
        if (!gst_tensor_query_client_update_caps (self, ret_str)) {
          nns_loge ("Failed to update client source caps.");
          ret = NNS_EDGE_ERROR_UNKNOWN;
        }
        g_free (ret_str);
      } else {
        /* respond deny with src caps string */
        nns_loge ("Query caps is not acceptable!");
        ret = NNS_EDGE_ERROR_UNKNOWN;
      }

      gst_caps_unref (server_caps);
      gst_caps_unref (client_caps);
      g_free (caps_str);
      break;
    }
    case NNS_EDGE_EVENT_NEW_DATA_RECEIVED:
    {
      nns_edge_data_h data;

      nns_edge_event_parse_new_data (event_h, &data);
      g_async_queue_push (self->msg_queue, data);
      break;
    }
    default:
      break;
  }

  return ret;
}

/**
 * @brief Internal function to create edge handle.
 */
static gboolean
gst_tensor_query_client_create_edge_handle (GstTensorQueryClient * self)
{
  gboolean started = FALSE;
  gchar *prev_caps = NULL;
  int ret;

  /* Already created, compare caps string. */
  if (self->edge_h) {
    ret = nns_edge_get_info (self->edge_h, "CAPS", &prev_caps);

    if (ret != NNS_EDGE_ERROR_NONE || !prev_caps ||
        !g_str_equal (prev_caps, self->in_caps_str)) {
      /* Capability is changed, close old handle. */
      nns_edge_release_handle (self->edge_h);
      self->edge_h = NULL;
    } else {
      return TRUE;
    }
  }

  ret = nns_edge_create_handle ("TEMP_ID", self->connect_type,
      NNS_EDGE_NODE_TYPE_QUERY_CLIENT, &self->edge_h);
  if (ret != NNS_EDGE_ERROR_NONE)
    return FALSE;

  nns_edge_set_event_callback (self->edge_h, _nns_edge_event_cb, self);

  if (self->topic)
    nns_edge_set_info (self->edge_h, "TOPIC", self->topic);
  if (self->host)
    nns_edge_set_info (self->edge_h, "HOST", self->host);
  if (self->port > 0) {
    gchar *port = g_strdup_printf ("%u", self->port);
    nns_edge_set_info (self->edge_h, "PORT", port);
    g_free (port);
  }
  nns_edge_set_info (self->edge_h, "CAPS", self->in_caps_str);

  ret = nns_edge_start (self->edge_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge
        ("Failed to start NNStreamer-edge. Please check server IP and port.");
    goto done;
  }

  ret = nns_edge_connect (self->edge_h, self->dest_host, self->dest_port);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to connect to edge server!");
    goto done;
  }

  started = TRUE;

done:
  if (!started) {
    nns_edge_release_handle (self->edge_h);
    self->edge_h = NULL;
  }

  return started;
}

/**
 * @brief This function handles sink event.
 */
static gboolean
gst_tensor_query_client_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (parent);

  GST_DEBUG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gboolean ret;

      gst_event_parse_caps (event, &caps);
      g_free (self->in_caps_str);
      self->in_caps_str = gst_caps_to_string (caps);

      ret = gst_tensor_query_client_create_edge_handle (self);
      if (!ret)
        nns_loge ("Failed to create edge handle, cannot start query client.");

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
  nns_edge_data_h data_h;
  guint i, num_mems, num_data;
  int ret;
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo map[NNS_TENSOR_SIZE_LIMIT];
  gchar *val;
  UNUSED (pad);

  ret = nns_edge_data_create (&data_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to create data handle in client chain.");
    return GST_FLOW_ERROR;
  }

  num_mems = gst_buffer_n_memory (buf);
  for (i = 0; i < num_mems; i++) {
    mem[i] = gst_buffer_peek_memory (buf, i);
    if (!gst_memory_map (mem[i], &map[i], GST_MAP_READ)) {
      ml_loge ("Cannot map the %uth memory in gst-buffer.", i);
      num_mems = i;
      goto done;
    }
    nns_edge_data_add (data_h, map[i].data, map[i].size, NULL);
  }

  nns_edge_get_info (self->edge_h, "client_id", &val);
  nns_edge_data_set_info (data_h, "client_id", val);
  g_free (val);

  if (NNS_EDGE_ERROR_NONE != nns_edge_send (self->edge_h, data_h)) {
    nns_logw ("Failed to publish to server node, retry connection.");
    goto retry;
  }

  nns_edge_data_destroy (data_h);

  data_h = g_async_queue_timeout_pop (self->msg_queue,
      self->timeout * G_TIME_SPAN_MILLISECOND);
  if (data_h) {
    ret = nns_edge_data_get_count (data_h, &num_data);
    if (ret != NNS_EDGE_ERROR_NONE || num_data == 0) {
      nns_loge ("Failed to get the number of memories of the edge data.");
      res = GST_FLOW_ERROR;
      goto done;
    }

    out_buf = gst_buffer_new ();
    for (i = 0; i < num_data; i++) {
      void *data = NULL;
      nns_size_t data_len;
      gpointer new_data;

      nns_edge_data_get (data_h, i, &data, &data_len);
      new_data = _g_memdup (data, data_len);
      gst_buffer_append_memory (out_buf,
          gst_memory_new_wrapped (0, new_data, data_len, 0,
              data_len, new_data, g_free));
    }
    /* metadata from incoming buffer */
    gst_buffer_copy_into (out_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

    res = gst_pad_push (self->srcpad, out_buf);
  }
  goto done;

retry:
  if (!self->topic || !_client_retry_connection (self)) {
    nns_loge ("Failed to retry connection");
    res = GST_FLOW_ERROR;
  }
done:
  if (data_h) {
    nns_edge_data_destroy (data_h);
  }

  for (i = 0; i < num_mems; i++)
    gst_memory_unmap (mem[i], &map[i]);

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
