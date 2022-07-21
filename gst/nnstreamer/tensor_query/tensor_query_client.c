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
#include "nnstreamer-edge.h"
#include "tensor_query_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <ifaddrs.h>

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
  PROP_PROTOCOL,
  PROP_OPERATION,
  PROP_SILENT,
};

#define TCP_HIGHEST_PORT        65535
#define TCP_DEFAULT_HOST        "localhost"
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
  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host",
          "A tenor query server host to send the packets to/from",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "The port of tensor query server to send the packets to/from", 0,
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
 * @brief Get IP address
 */
static gchar *
_get_ip_address (void)
{
  struct ifaddrs *addrs, *run;
  gchar *ret = NULL;

  if (0 != getifaddrs (&addrs))
    goto done;
  run = addrs;

  while (run) {
    if (run->ifa_addr && run->ifa_addr->sa_family == AF_INET) {
      struct sockaddr_in *pAddr = (struct sockaddr_in *) run->ifa_addr;

      if (NULL != strstr (run->ifa_name, "en") ||
          NULL != strstr (run->ifa_name, "et")) {
        g_free (ret);
        ret = g_strdup (inet_ntoa (pAddr->sin_addr));
      }
    }
    run = run->ifa_next;
  }
  freeifaddrs (addrs);

done:
  if (NULL == ret)
    ret = g_strdup ("localhost");

  return ret;
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
  self->host = g_strdup (TCP_DEFAULT_HOST);
  self->port = TCP_DEFAULT_SRC_PORT;
  self->srv_host = g_strdup (TCP_DEFAULT_HOST);
  self->srv_port = TCP_DEFAULT_SRC_PORT;
  self->operation = NULL;
  self->in_caps_str = NULL;
  self->msg_queue = g_async_queue_new ();

  tensor_query_hybrid_init (&self->hybrid_info, NULL, 0, FALSE);
  nns_edge_create_handle ("TEMP_ID", "TEMP_CLIENT_TOPIC", &self->edge_h);
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
  g_free (self->srv_host);
  self->srv_host = NULL;
  g_free (self->operation);
  self->operation = NULL;
  g_free (self->in_caps_str);
  self->in_caps_str = NULL;

  while ((data_h = g_async_queue_try_pop (self->msg_queue))) {
    nns_edge_data_destroy (data_h);
  }
  g_async_queue_unref (self->msg_queue);

  tensor_query_hybrid_close (&self->hybrid_info);
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
    case PROP_PROTOCOL:
    {
        /** @todo expand when other protocols are ready */
      nns_edge_protocol_e protocol = g_value_get_enum (value);
      if (protocol == NNS_EDGE_PROTOCOL_TCP)
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
    case PROP_PROTOCOL:
      g_value_set_enum (value, self->protocol);
      break;
    case PROP_OPERATION:
      g_value_set_string (value, self->operation);
      break;
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
 * @brief Copy server info.
 */
static void
_copy_srv_info (GstTensorQueryClient * self, query_server_info_s * server)
{
  g_free (self->srv_host);
  self->srv_host = g_strdup (server->src.host);
  self->srv_port = server->src.port;
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
    nns_edge_disconnect (self->edge_h);

    _copy_srv_info (self, server);
    tensor_query_hybrid_free_server_info (server);

    if (nns_edge_connect (self->edge_h, NNS_EDGE_PROTOCOL_TCP, self->srv_host,
            self->srv_port)) {
      nns_logi ("Connected to new server: %s:%u.", self->srv_host,
          self->srv_port);
      ret = TRUE;
      break;
    }
  }

  return ret;
}

/**
 * @brief Parse caps from received event data.
 */
static gchar *
_nns_edge_parse_caps (gchar * caps_str, gboolean is_src)
{
  gchar **strv = g_strsplit (caps_str, "@", -1);
  gint num = g_strv_length (strv), i;
  gchar *find_key = NULL;
  gchar *ret_str = NULL;

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
  gchar *caps_str = NULL;
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
      gchar *ret_str;

      nns_edge_event_parse_capability (event_h, &caps_str);
      ret_str = _nns_edge_parse_caps (caps_str, TRUE);
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

  g_free (caps_str);
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
  gchar *ip, *prev_caps_str, *new_caps_str;

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
            self->host, self->port);

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
        g_free (self->srv_host);
        self->srv_host = g_strdup (self->host);
        self->srv_port = self->port;
      }

      g_free (self->in_caps_str);
      self->in_caps_str = gst_caps_to_string (caps);
      nns_edge_get_info (self->edge_h, "CAPS", &prev_caps_str);
      if (!prev_caps_str)
        prev_caps_str = g_strdup ("");
      new_caps_str = g_strconcat (prev_caps_str, self->in_caps_str, NULL);
      nns_edge_set_info (self->edge_h, "CAPS", new_caps_str);
      g_free (prev_caps_str);
      g_free (new_caps_str);

      nns_edge_set_event_callback (self->edge_h, _nns_edge_event_cb, self);

      ip = _get_ip_address ();
      nns_edge_set_info (self->edge_h, "IP", ip);
      nns_edge_set_info (self->edge_h, "PORT", "0");
      g_free (ip);

      if (0 != nns_edge_start (self->edge_h, false)) {
        nns_loge
            ("Failed to start NNStreamer-edge. Please check server IP and port");
        return FALSE;
      }

      if (0 != nns_edge_connect (self->edge_h, NNS_EDGE_PROTOCOL_TCP,
              self->srv_host, self->srv_port)) {
        nns_loge ("Failed to connect to edge server!");
        return FALSE;
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
  nns_edge_data_h data_h;
  guint i, num_mems, num_data;
  int ret;
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo map[NNS_TENSOR_SIZE_LIMIT];
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
  if (0 != nns_edge_request (self->edge_h, data_h)) {
    nns_logw ("Failed to request to server node, retry connection.");
    goto retry;
  }

  nns_edge_data_destroy (data_h);

  data_h = g_async_queue_try_pop (self->msg_queue);
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
      size_t data_len;
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
  if (!self->operation || !_client_retry_connection (self)) {
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
