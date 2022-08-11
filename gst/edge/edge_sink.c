/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_sink.c
 * @date    01 Aug 2022
 * @brief   Publish incoming streams
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "edge_sink.h"

GST_DEBUG_CATEGORY_STATIC (gst_edgesink_debug);
#define GST_CAT_DEFAULT gst_edgesink_debug

/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

/**
 * @brief edgesink properties
 */
enum
{
  PROP_0,

  PROP_HOST,
  PROP_PORT,
  PROP_CONNECT_TYPE,

  PROP_LAST
};

#define gst_edgesink_parent_class parent_class
G_DEFINE_TYPE (GstEdgeSink, gst_edgesink, GST_TYPE_BASE_SINK);

static void gst_edgesink_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);

static void gst_edgesink_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

static void gst_edgesink_finalize (GObject * object);

static gboolean gst_edgesink_start (GstBaseSink * basesink);
static GstFlowReturn gst_edgesink_render (GstBaseSink * basesink,
    GstBuffer * buffer);
static gboolean gst_edgesink_set_caps (GstBaseSink * basesink, GstCaps * caps);

static gchar *gst_edgesink_get_host (GstEdgeSink * self);
static void gst_edgesink_set_host (GstEdgeSink * self, const gchar * host);

static guint16 gst_edgesink_get_port (GstEdgeSink * self);
static void gst_edgesink_set_port (GstEdgeSink * self, const guint16 port);

static nns_edge_connect_type_e gst_edgesink_get_connect_type (GstEdgeSink *
    self);
static void gst_edgesink_set_connect_type (GstEdgeSink * self,
    const nns_edge_connect_type_e connect_type);

/**
 * @brief initialize the class
 */
static void
gst_edgesink_class_init (GstEdgeSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gstbasesink_class = (GstBaseSinkClass *) klass;
  gstelement_class = (GstElementClass *) gstbasesink_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_edgesink_set_property;
  gobject_class->get_property = gst_edgesink_get_property;
  gobject_class->finalize = gst_edgesink_finalize;

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host",
          "A self host address to accept connection from edgesrc", DEFAULT_HOST,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "A self port address to accept connection from edgesrc.",
          0, 65535, DEFAULT_PORT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CONNECT_TYPE,
      g_param_spec_enum ("connect-type", "Connect Type",
          "The connections type between edgesink and edgesrc.",
          GST_TYPE_EDGE_CONNECT_TYPE, DEFAULT_CONNECT_TYPE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "EdgeSink", "Sink/Edge",
      "Publish incoming streams", "Samsung Electronics Co., Ltd.");

  gstbasesink_class->start = gst_edgesink_start;
  gstbasesink_class->render = gst_edgesink_render;
  gstbasesink_class->set_caps = gst_edgesink_set_caps;

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
      GST_EDGE_ELEM_NAME_SINK, 0, "Edge sink");
}

/**
 * @brief initialize the new element
 */
static void
gst_edgesink_init (GstEdgeSink * self)
{
  self->host = g_strdup (DEFAULT_HOST);
  self->port = DEFAULT_PORT;
  self->connect_type = DEFAULT_CONNECT_TYPE;
}

/**
 * @brief set property
 */
static void
gst_edgesink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstEdgeSink *self = GST_EDGESINK (object);

  switch (prop_id) {
    case PROP_HOST:
      gst_edgesink_set_host (self, g_value_get_string (value));
      break;
    case PROP_PORT:
      gst_edgesink_set_port (self, g_value_get_uint (value));
      break;
    case PROP_CONNECT_TYPE:
      gst_edgesink_set_connect_type (self, g_value_get_enum (value));
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
gst_edgesink_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstEdgeSink *self = GST_EDGESINK (object);

  switch (prop_id) {
    case PROP_HOST:
      g_value_set_string (value, gst_edgesink_get_host (self));
      break;
    case PROP_PORT:
      g_value_set_uint (value, gst_edgesink_get_port (self));
      break;
    case PROP_CONNECT_TYPE:
      g_value_set_enum (value, gst_edgesink_get_connect_type (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the object
 */
static void
gst_edgesink_finalize (GObject * object)
{
  GstEdgeSink *self = GST_EDGESINK (object);
  if (self->host) {
    g_free (self->host);
    self->host = NULL;
  }

  if (self->edge_h) {
    nns_edge_release_handle (self->edge_h);
    self->edge_h = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief start processing of edgesink
 */
static gboolean
gst_edgesink_start (GstBaseSink * basesink)
{
  GstEdgeSink *self = GST_EDGESINK (basesink);

  int ret;
  char *port = NULL;

  ret =
      nns_edge_create_handle ("TEMP_ID", self->connect_type,
      NNS_EDGE_NODE_TYPE_PUB, &self->edge_h);

  if (NNS_EDGE_ERROR_NONE != ret) {
    nns_loge ("Failed to get nnstreamer edge handle.");

    if (self->edge_h) {
      nns_edge_release_handle (self->edge_h);
      self->edge_h = NULL;
    }

    return FALSE;
  }

  nns_edge_set_info (self->edge_h, "HOST", self->host);
  port = g_strdup_printf ("%d", self->port);
  nns_edge_set_info (self->edge_h, "PORT", port);
  g_free (port);

  if (0 != nns_edge_start (self->edge_h)) {
    nns_loge
        ("Failed to start NNStreamer-edge. Please check server IP and port");
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief render buffer, send buffer
 */
static GstFlowReturn
gst_edgesink_render (GstBaseSink * basesink, GstBuffer * buffer)
{
  GstEdgeSink *self = GST_EDGESINK (basesink);
  nns_edge_data_h data_h;
  guint i, num_mems;
  int ret;
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo map[NNS_TENSOR_SIZE_LIMIT];

  ret = nns_edge_data_create (&data_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_loge ("Failed to create data handle in edgesink");
    return GST_FLOW_ERROR;
  }

  num_mems = gst_buffer_n_memory (buffer);
  for (i = 0; i < num_mems; i++) {
    mem[i] = gst_buffer_peek_memory (buffer, i);
    if (!gst_memory_map (mem[i], &map[i], GST_MAP_READ)) {
      nns_loge ("Cannot map the %uth memory in gst-buffer", i);
      num_mems = i;
      goto done;
    }
    nns_edge_data_add (data_h, map[i].data, map[i].size, NULL);
  }

  nns_edge_send (self->edge_h, data_h);
  goto done;

done:
  if (data_h)
    nns_edge_data_destroy (data_h);

  for (i = 0; i < num_mems; i++) {
    gst_memory_unmap (mem[i], &map[i]);
  }

  return GST_FLOW_OK;
}

/**
 * @brief An implementation of the set_caps vmethod in GstBaseSinkClass
 */
static gboolean
gst_edgesink_set_caps (GstBaseSink * basesink, GstCaps * caps)
{
  GstEdgeSink *sink = GST_EDGESINK (basesink);
  gchar *caps_str, *prev_caps_str, *new_caps_str;
  int set_rst;

  caps_str = gst_caps_to_string (caps);

  nns_edge_get_info (sink->edge_h, "CAPS", &prev_caps_str);
  if (!prev_caps_str) {
    prev_caps_str = g_strdup ("");
  }
  new_caps_str =
      g_strdup_printf ("%s@edge_sink_caps@%s", prev_caps_str, caps_str);
  set_rst = nns_edge_set_info (sink->edge_h, "CAPS", new_caps_str);

  g_free (prev_caps_str);
  g_free (new_caps_str);
  g_free (caps_str);

  return set_rst == NNS_EDGE_ERROR_NONE;
}

/**
 * @brief getter for the 'host' property.
 */
static gchar *
gst_edgesink_get_host (GstEdgeSink * self)
{
  return self->host;
}

/**
 * @brief setter for the 'host' property.
 */
static void
gst_edgesink_set_host (GstEdgeSink * self, const gchar * host)
{
  if (self->host)
    g_free (self->host);
  self->host = g_strdup (host);
}

/**
 * @brief getter for the 'port' property.
 */
static guint16
gst_edgesink_get_port (GstEdgeSink * self)
{
  return self->port;
}

/**
 * @brief setter for the 'port' property.
 */
static void
gst_edgesink_set_port (GstEdgeSink * self, const guint16 port)
{
  self->port = port;
}

/**
 * @brief getter for the 'connect_type' property.
 */
static nns_edge_connect_type_e
gst_edgesink_get_connect_type (GstEdgeSink * self)
{
  return self->connect_type;
}

/**
 * @brief setter for the 'connect_type' property.
 */
static void
gst_edgesink_set_connect_type (GstEdgeSink * self,
    const nns_edge_connect_type_e connect_type)
{
  self->connect_type = connect_type;
}
