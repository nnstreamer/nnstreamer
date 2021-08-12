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

#include "tensor_query_client.h"
#include <gio/gio.h>
#include <gio/gsocket.h>

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
  PROP_SILENT,
};

#define TCP_HIGHEST_PORT        65535
#define TCP_DEFAULT_HOST        "localhost"
#define TCP_DEFAULT_SINK_PORT        3000
#define TCP_DEFAULT_SRC_PORT        3001
#define DEFAULT_SILENT TRUE
#define DEFAULT_PROTOCOL        "tcp"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_client_debug);
#define GST_CAT_DEFAULT gst_tensor_query_client_debug

/**
 * @brief Default caps string for pads.
 */
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_DEFAULT ";" GST_TENSORS_FLEX_CAP_DEFAULT

/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief the capabilities of the outputs.
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

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
      g_param_spec_string ("sink-host", "Host",
          "A tenor query sink host to send the packets to/from",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SINK_PORT,
      g_param_spec_int ("sink-port", "Port",
          "The port of tensor query sink to send the packets to/from", 0,
          TCP_HIGHEST_PORT, TCP_DEFAULT_SINK_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SRC_HOST,
      g_param_spec_string ("src-host", "Host",
          "A tenor query src host to send the packets to/from",
          TCP_DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SRC_PORT,
      g_param_spec_int ("src-port", "Port",
          "The port of tensor query src to send the packets to/from", 0,
          TCP_HIGHEST_PORT, TCP_DEFAULT_SRC_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PROTOCOL,
      g_param_spec_string ("protocol", "Protocol",
          "A protocol option for tensor query.",
          DEFAULT_PROTOCOL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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

/** @todo Remove when the dummy functions are implemented. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

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
  self->protocol = _TENSOR_QUERY_PROTOCOL_TCP;
  self->sink_conn = NULL;
  self->sink_host = g_strdup (TCP_DEFAULT_HOST);
  self->sink_port = TCP_DEFAULT_SINK_PORT;
  self->src_conn = NULL;
  self->src_host = g_strdup (TCP_DEFAULT_HOST);
  self->src_port = TCP_DEFAULT_SRC_PORT;

  gst_tensors_config_init (&self->in_config);
  gst_tensors_config_init (&self->out_config);
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

  nnstreamer_query_close (self->sink_conn);
  nnstreamer_query_close (self->src_conn);

  gst_tensors_config_free (&self->in_config);
  gst_tensors_config_free (&self->out_config);

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
        g_warning ("host property cannot be NULL");
        break;
      }
      g_free (self->sink_host);
      self->sink_host = g_value_dup_string (value);
      break;
    case PROP_SINK_PORT:
      self->sink_port = g_value_get_int (value);
      break;
    case PROP_SRC_HOST:
      if (!g_value_get_string (value)) {
        g_warning ("host property cannot be NULL");
        break;
      }
      g_free (self->src_host);
      self->src_host = g_value_dup_string (value);
      break;
    case PROP_SRC_PORT:
      self->src_port = g_value_get_int (value);
      break;
    case PROP_PROTOCOL:
      if (g_ascii_strcasecmp (g_value_get_string (value), "tcp") == 0)
        self->protocol = _TENSOR_QUERY_PROTOCOL_TCP;
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
      g_value_set_int (value, self->sink_port);
      break;
    case PROP_SRC_HOST:
      g_value_set_string (value, self->src_host);
      break;
    case PROP_SRC_PORT:
      g_value_set_int (value, self->src_port);
      break;
    case PROP_PROTOCOL:
      switch (self->protocol) {
        case _TENSOR_QUERY_PROTOCOL_TCP:
          g_value_set_string (value, "tcp");
          break;
        default:
          break;
      }
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
static void
gst_tensor_query_client_update_caps (GstTensorQueryClient * self)
{
  GstTensorsConfig *config;
  GstCaps *curr_caps, *out_caps;

  config = &self->out_config;
  out_caps = gst_tensor_pad_caps_from_config (self->srcpad, config);

  /* Update src pad caps if it is different. */
  curr_caps = gst_pad_get_current_caps (self->srcpad);
  if (curr_caps == NULL || !gst_caps_is_equal (curr_caps, out_caps)) {
    silent_debug_caps (self, out_caps, "set out-caps");
    gst_pad_set_caps (self->srcpad, out_caps);
  }

  if (curr_caps)
    gst_caps_unref (curr_caps);

  gst_caps_unref (out_caps);
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
      GstStructure *structure;

      gst_event_parse_caps (event, &caps);
      structure = gst_caps_get_structure (caps, 0);
      gst_tensors_config_from_structure (&self->in_config, structure);
      gst_event_unref (event);

      if (gst_tensors_config_validate (&self->in_config)) {
        /** Request out config to server_src. */
        TensorQueryCommandData cmd_buf;

        self->src_conn =
            nnstreamer_query_connect (self->protocol, self->src_host,
            self->src_port, DEFAULT_TIMEOUT_MS);
        if (!self->src_conn) {
          nns_loge ("Failed to connect server source ");
          return FALSE;
        }

        cmd_buf.cmd = _TENSOR_QUERY_CMD_REQUEST_INFO;
        cmd_buf.protocol = self->protocol;
        gst_tensors_config_copy (&cmd_buf.data_info.config, &self->in_config);

        if (0 != nnstreamer_query_send (self->src_conn, &cmd_buf,
                DEFAULT_TIMEOUT_MS)) {
          nns_loge ("Failed to send request info cmd buf");
          return FALSE;
        }

        if (0 != nnstreamer_query_receive (self->src_conn, &cmd_buf,
                DEFAULT_TIMEOUT_MS)) {
          nns_loge ("Failed to receive response from the query server.");
          return FALSE;
        }

        if (cmd_buf.cmd == _TENSOR_QUERY_CMD_RESPOND_APPROVE) {
          if (gst_tensors_config_validate (&cmd_buf.data_info.config)) {
            gst_tensors_info_copy (&self->out_config.info,
                &cmd_buf.data_info.config.info);
            /** The server's framerate is 0/1, set it the same as the input. */
            self->out_config.format = cmd_buf.data_info.config.format;
            self->out_config.rate_n = self->in_config.rate_n;
            self->out_config.rate_d = self->in_config.rate_d;
            gst_tensor_query_client_update_caps (self);
          }
        } else {
          /** @todo Retry for info */
          nns_loge ("Failed to receive approve command.");
          return FALSE;
        }
      }

      self->sink_conn =
          nnstreamer_query_connect (self->protocol, self->sink_host,
          self->sink_port, DEFAULT_TIMEOUT_MS);
      if (!self->sink_conn) {
        nns_loge ("Failed to connect server sink ");
        return FALSE;
      }

      return TRUE;
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
 * @brief Get start command buffer
 */
static gboolean
gst_tensor_query_client_handle_cmd_buf (GstTensorQueryClient * self,
    GstBuffer * buf, TensorQueryCommandData * cmd_buf)
{
  GstMemory *in_mem;
  GstMapInfo in_info;
  guint i, num_tensors;

  num_tensors = gst_buffer_n_memory (buf);

  if (cmd_buf->cmd == _TENSOR_QUERY_CMD_TRANSFER_START) {
    cmd_buf->data_info.base_time =
        gst_element_get_base_time (GST_ELEMENT (self));
    cmd_buf->data_info.duration = GST_BUFFER_DURATION (buf);
    cmd_buf->data_info.dts = GST_BUFFER_DTS (buf);
    cmd_buf->data_info.pts = GST_BUFFER_PTS (buf);
    cmd_buf->data_info.num_mems = num_tensors;
  }

  for (i = 0; i < num_tensors; i++) {
    in_mem = gst_buffer_peek_memory (buf, i);
    if (!gst_memory_map (in_mem, &in_info, GST_MAP_READ)) {
      nns_loge ("Cannot map input memory / tensor query client.");
      return FALSE;
    }

    if (cmd_buf->cmd == _TENSOR_QUERY_CMD_TRANSFER_START) {
      cmd_buf->data_info.mem_sizes[i] = in_info.size;
    } else if (cmd_buf->cmd == _TENSOR_QUERY_CMD_TRANSFER_DATA) {
      cmd_buf->data.data = in_info.data;
      cmd_buf->data.size = in_info.size;
      if (0 != nnstreamer_query_send (self->src_conn, cmd_buf,
              DEFAULT_TIMEOUT_MS)) {
        gst_memory_unmap (in_mem, &in_info);
        nns_loge ("Failed to send %uth data command buffer", i);
        return FALSE;
      }
    } else {
      gst_memory_unmap (in_mem, &in_info);
      nns_loge ("Undefiend behavior with this command: %d", cmd_buf->cmd);
      return FALSE;
    }
    gst_memory_unmap (in_mem, &in_info);
  }

  return TRUE;
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_tensor_query_client_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buf)
{
  GstTensorQueryClient *self = GST_TENSOR_QUERY_CLIENT (parent);
  TensorQueryCommandData cmd_buf;
  guint i, num_tensors = 0;
  guint mem_sizes[NNS_TENSOR_SIZE_LIMIT];
  GstBuffer *out_buf = NULL;
  GstMemory *out_mem;
  GstMapInfo out_info;
  GstFlowReturn res = GST_FLOW_OK;
  gint ecode;
  gboolean is_flexible = gst_tensors_config_is_flexible (&self->out_config);

  /** Send start command buffer */
  cmd_buf.protocol = self->protocol;
  cmd_buf.cmd = _TENSOR_QUERY_CMD_TRANSFER_START;
  if (!gst_tensor_query_client_handle_cmd_buf (self, buf, &cmd_buf)) {
    nns_loge ("Failed to hanlde %d command buffers.", cmd_buf.cmd);
    return GST_FLOW_ERROR;
  }
  if (0 != nnstreamer_query_send (self->src_conn, &cmd_buf, DEFAULT_TIMEOUT_MS)) {
    nns_loge ("Failed to send start command buffer");
    return GST_FLOW_ERROR;
  }

  /** Send data command buffer */
  cmd_buf.cmd = _TENSOR_QUERY_CMD_TRANSFER_DATA;
  if (!gst_tensor_query_client_handle_cmd_buf (self, buf, &cmd_buf)) {
    nns_loge ("Failed to hanlde %d command buffers.", cmd_buf.cmd);
    return GST_FLOW_ERROR;
  }

  /** Send end command buffer */
  cmd_buf.cmd = _TENSOR_QUERY_CMD_TRANSFER_END;
  if (0 != nnstreamer_query_send (self->src_conn, &cmd_buf, DEFAULT_TIMEOUT_MS)) {
    nns_loge ("Failed to send end command buffer");
    return GST_FLOW_ERROR;
  }
  /** Receive start command buffer */
  if (0 != nnstreamer_query_receive (self->sink_conn, &cmd_buf,
          DEFAULT_TIMEOUT_MS)) {
    nns_loge ("Failed to receive start command buffer");
    return GST_FLOW_ERROR;
  }

  if (cmd_buf.cmd == _TENSOR_QUERY_CMD_TRANSFER_START) {
    num_tensors = cmd_buf.data_info.num_mems;

    if (!is_flexible && num_tensors != self->out_config.info.num_tensors) {
      nns_loge
          ("The number of tensors to receive does not match with out config.");
      return GST_FLOW_ERROR;
    }
    for (i = 0; i < num_tensors; i++) {
      mem_sizes[i] = cmd_buf.data_info.mem_sizes[i];
      if (!is_flexible && mem_sizes[i] !=
          gst_tensor_info_get_size (&self->out_config.info.info[i])) {
        nns_loge
            ("Size of the tensor to receive does not match with out config.");
        return GST_FLOW_ERROR;
      }
    }
  }

  out_buf = gst_buffer_new ();

  /** Receive data command buffer */
  for (i = 0; i < num_tensors; i++) {
    out_mem = gst_allocator_alloc (NULL, mem_sizes[i], NULL);
    gst_buffer_append_memory (out_buf, out_mem);

    if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
      nns_loge ("Cannot map gst memory (query-client buffer)");
      goto error;
    }
    cmd_buf.data.data = out_info.data;

    ecode = nnstreamer_query_receive (self->sink_conn, &cmd_buf,
        DEFAULT_TIMEOUT_MS);
    gst_memory_unmap (out_mem, &out_info);

    if (ecode != 0) {
      nns_loge ("Failed to receive %u th data command buffer", i);
      goto error;
    }
  }

  /** Receive end command buffer */
  if (0 != nnstreamer_query_receive (self->sink_conn, &cmd_buf,
          DEFAULT_TIMEOUT_MS)) {
    nns_loge ("Failed to receive end command buffer");
    goto error;
  }
  if (cmd_buf.cmd != _TENSOR_QUERY_CMD_TRANSFER_END) {
    nns_loge ("Expected _TENSOR_QUERY_CMD_TRANSFER_END, but received %d.",
        cmd_buf.cmd);
    goto error;
  }

  out_buf = gst_buffer_make_writable (out_buf);

  /* metadata from incoming buffer */
  gst_buffer_copy_into (out_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

  res = gst_pad_push (self->srcpad, out_buf);

  goto done;

error:
  res = GST_FLOW_ERROR;
  gst_buffer_unref (out_buf);
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

#pragma GCC diagnostic pop
