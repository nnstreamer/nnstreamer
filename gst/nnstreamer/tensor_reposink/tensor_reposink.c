/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */

/**
 * SECTION: element-tensor_reposink
 *
 * Set elemnt to handle tensor repo
 *
 * @file	tensor_reposink.c
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_repo.h"
#include "tensor_reposink.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_reposink_debug);
#define GST_CAT_DEFAULT gst_tensor_reposink_debug

/**
 * @brief tensor_reposink properties
 */
enum
{
  PROP_0,
  PROP_SIGNAL_RATE,
  PROP_SLOT,
  PROP_SILENT
};

#define DEFAULT_SIGNAL_RATE 0
#define DEFAULT_SILENT TRUE
#define DEFAULT_QOS TRUE
#define DEFAULT_INDEX 0

/**
 * @brief tensor_reposink sink template
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

static void gst_tensor_reposink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_reposink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_reposink_dispose (GObject * object);

static gboolean gst_tensor_reposink_start (GstBaseSink * sink);
static gboolean gst_tensor_reposink_stop (GstBaseSink * sink);
static gboolean gst_tensor_reposink_event (GstBaseSink * sink,
    GstEvent * event);
static gboolean gst_tensor_reposink_query (GstBaseSink * sink,
    GstQuery * query);
static GstFlowReturn gst_tensor_reposink_render (GstBaseSink * sink,
    GstBuffer * buffer);
static GstFlowReturn gst_tensor_reposink_render_list (GstBaseSink * sink,
    GstBufferList * buffer_list);
static gboolean gst_tensor_reposink_set_caps (GstBaseSink * sink,
    GstCaps * caps);
static GstCaps *gst_tensor_reposink_get_caps (GstBaseSink * sink,
    GstCaps * filter);


#define gst_tensor_reposink_parent_class parent_class
G_DEFINE_TYPE (GstTensorRepoSink, gst_tensor_reposink, GST_TYPE_BASE_SINK);

/**
 * @brief class initialization of tensor_reposink
 */
static void
gst_tensor_reposink_class_init (GstTensorRepoSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstBaseSinkClass *basesink_class;
  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);
  basesink_class = GST_BASE_SINK_CLASS (klass);

  gobject_class->set_property = gst_tensor_reposink_set_property;
  gobject_class->get_property = gst_tensor_reposink_get_property;
  gobject_class->dispose = gst_tensor_reposink_dispose;

  g_object_class_install_property (gobject_class, PROP_SIGNAL_RATE,
      g_param_spec_uint ("signal-rate", "Signal rate",
          "New data signals per second (0 for unlimited, max 500)", 0, 500,
          DEFAULT_SIGNAL_RATE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SLOT,
      g_param_spec_uint ("slot-index", "Slot Index", "repository slot index",
          0, UINT_MAX, DEFAULT_INDEX,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "TensorRepoSink",
      "Set/TensorRepo",
      "Set element to handle tensor repository",
      "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (element_class, &sink_template);

  basesink_class->start = GST_DEBUG_FUNCPTR (gst_tensor_reposink_start);
  basesink_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_reposink_stop);
  basesink_class->event = GST_DEBUG_FUNCPTR (gst_tensor_reposink_event);
  basesink_class->query = GST_DEBUG_FUNCPTR (gst_tensor_reposink_query);
  basesink_class->render = GST_DEBUG_FUNCPTR (gst_tensor_reposink_render);
  basesink_class->render_list =
      GST_DEBUG_FUNCPTR (gst_tensor_reposink_render_list);
  basesink_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_reposink_set_caps);
  basesink_class->get_caps = GST_DEBUG_FUNCPTR (gst_tensor_reposink_get_caps);
}

/**
 * @brief initialization of tensor_reposink
 */
static void
gst_tensor_reposink_init (GstTensorRepoSink * self)
{
  GstBaseSink *basesink;
  basesink = GST_BASE_SINK (self);

  gst_tensor_repo_init ();

  GST_DEBUG_OBJECT (self, "GstTensorRepo is sucessfully initailzed");

  self->silent = DEFAULT_SILENT;
  self->signal_rate = DEFAULT_SIGNAL_RATE;
  self->last_render_time = GST_CLOCK_TIME_NONE;
  self->set_startid = FALSE;
  self->in_caps = NULL;

  gst_base_sink_set_qos_enabled (basesink, DEFAULT_QOS);
}

/**
 * @brief set property vmethod
 */
static void
gst_tensor_reposink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorRepoSink *self;
  self = GST_TENSOR_REPOSINK (object);

  switch (prop_id) {
    case PROP_SIGNAL_RATE:
      self->signal_rate = g_value_get_uint (value);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_SLOT:
      self->o_myid = self->myid;
      self->myid = g_value_get_uint (value);
      gst_tensor_repo_add_repodata (self->myid, TRUE);
      if (!self->set_startid) {
        self->o_myid = self->myid;
        self->set_startid = TRUE;
      }
      if (self->o_myid != self->myid)
        gst_tensor_repo_set_changed (self->o_myid, self->myid, TRUE);

      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property vmethod
 */
static void
gst_tensor_reposink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorRepoSink *self;
  self = GST_TENSOR_REPOSINK (object);
  switch (prop_id) {
    case PROP_SIGNAL_RATE:
      g_value_set_uint (value, self->signal_rate);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_SLOT:
      g_value_set_uint (value, self->myid);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief dispose vmethod implementation
 */
static void
gst_tensor_reposink_dispose (GObject * object)
{
  GstTensorRepoSink *self;
  self = GST_TENSOR_REPOSINK (object);
  if (self->in_caps)
    gst_caps_unref (self->in_caps);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief start vmethod implementation
 */
static gboolean
gst_tensor_reposink_start (GstBaseSink * sink)
{
  return TRUE;
}

/**
 * @brief stop vmethod implementation
 */
static gboolean
gst_tensor_reposink_stop (GstBaseSink * sink)
{
  return TRUE;
}

/**
 * @brief Handle events.
 *
 * GstBaseSink method implementation.
 */
static gboolean
gst_tensor_reposink_event (GstBaseSink * sink, GstEvent * event)
{
  GstTensorRepoSink *self;
  GstEventType type;

  self = GST_TENSOR_REPOSINK (sink);
  type = GST_EVENT_TYPE (event);

  GST_DEBUG_OBJECT (self, "received event %s", GST_EVENT_TYPE_NAME (event));

  switch (type) {
    case GST_EVENT_EOS:
      gst_tensor_repo_set_eos (self->myid);
      break;

    default:
      break;
  }
  return GST_BASE_SINK_CLASS (parent_class)->event (sink, event);
}

/**
 * @brief query vmethod implementation
 */
static gboolean
gst_tensor_reposink_query (GstBaseSink * sink, GstQuery * query)
{
  GstTensorRepoSink *self;
  GstQueryType type;
  GstFormat format;

  self = GST_TENSOR_REPOSINK (sink);
  type = GST_QUERY_TYPE (query);

  GST_DEBUG_OBJECT (self, "received query %s", GST_QUERY_TYPE_NAME (query));
  switch (type) {
    case GST_QUERY_SEEKING:
      gst_query_parse_seeking (query, &format, NULL, NULL, NULL);
      gst_query_set_seeking (query, format, FALSE, 0, -1);
      return TRUE;

    default:
      break;
  }

  return GST_BASE_SINK_CLASS (parent_class)->query (sink, query);
}


/**
 * @brief Push GstBuffer
 */
static void
gst_tensor_reposink_render_buffer (GstTensorRepoSink * self, GstBuffer * buffer)
{
  GstClockTime now = GST_CLOCK_TIME_NONE;
  guint signal_rate;
  gboolean notify = FALSE;
  g_return_if_fail (GST_IS_TENSOR_REPOSINK (self));

  signal_rate = self->signal_rate;

  if (signal_rate) {
    GstClock *clock;
    GstClockTime render_time;
    clock = gst_element_get_clock (GST_ELEMENT (self));

    if (clock) {
      now = gst_clock_get_time (clock);
      render_time = (1000 / signal_rate) * GST_MSECOND + self->last_render_time;
      if (!GST_CLOCK_TIME_IS_VALID (self->last_render_time) ||
          GST_CLOCK_DIFF (now, render_time) <= 0)
        notify = TRUE;
      gst_object_unref (clock);
    }
  } else {
    notify = TRUE;
  }

  if (notify) {
    gboolean ret = FALSE;
    self->last_render_time = now;
    ret =
        gst_tensor_repo_set_buffer (self->myid, self->o_myid, buffer,
        self->in_caps);
    if (!ret)
      GST_ELEMENT_ERROR (self, RESOURCE, WRITE,
          ("Cannot Set buffer into repo [key: %d]", self->myid), NULL);
  }
}

/**
 * @brief render vmethod implementation
 */
static GstFlowReturn
gst_tensor_reposink_render (GstBaseSink * sink, GstBuffer * buffer)
{
  GstTensorRepoSink *self;
  self = GST_TENSOR_REPOSINK (sink);


  gst_tensor_reposink_render_buffer (self, buffer);
  return GST_FLOW_OK;
}

/**
 * @brief render list vmethod implementation
 */
static GstFlowReturn
gst_tensor_reposink_render_list (GstBaseSink * sink,
    GstBufferList * buffer_list)
{
  GstTensorRepoSink *self;
  GstBuffer *buffer;
  guint i;
  guint num_buffers;

  self = GST_TENSOR_REPOSINK (sink);
  num_buffers = gst_buffer_list_length (buffer_list);

  for (i = 0; i < num_buffers; i++) {
    buffer = gst_buffer_list_get (buffer_list, i);
    gst_tensor_reposink_render_buffer (self, buffer);
  }

  return GST_FLOW_OK;
}

/**
 * @brief set_caps vmethod implementation
 */
static gboolean
gst_tensor_reposink_set_caps (GstBaseSink * sink, GstCaps * caps)
{
  GstTensorRepoSink *self;

  self = GST_TENSOR_REPOSINK (sink);
  gst_caps_replace (&self->in_caps, caps);

  if (!self->silent) {
    guint caps_size, i;

    caps_size = gst_caps_get_size (caps);
    GST_DEBUG_OBJECT (self, "set caps, size is %d", caps_size);

    for (i = 0; i < caps_size; i++) {
      GstStructure *structure = gst_caps_get_structure (caps, i);
      gchar *str = gst_structure_to_string (structure);

      GST_DEBUG_OBJECT (self, "[%d] %s", i, str);
      g_free (str);
    }
  }

  return TRUE;
}

/**
 * @brief get_caps vmethod implementation
 */
static GstCaps *
gst_tensor_reposink_get_caps (GstBaseSink * sink, GstCaps * filter)
{
  GstTensorRepoSink *self;
  GstCaps *caps;

  self = GST_TENSOR_REPOSINK (sink);

  caps = self->in_caps;

  if (caps) {
    if (filter) {
      caps = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
    } else {
      gst_caps_ref (caps);
    }
  }

  return caps;
}


/**
 * @brief Function to initialize the plugin.
 *
 * See GstPluginInitFunc() for more details.
 */
NNSTREAMER_PLUGIN_INIT (tensor_reposink)
{
  GST_DEBUG_CATEGORY_INIT (gst_tensor_reposink_debug, "tensor_reposink",
      0, "tensor_reposink element");

  return gst_element_register (plugin, "tensor_reposink",
      GST_RANK_NONE, GST_TYPE_TENSOR_REPOSINK);
}
