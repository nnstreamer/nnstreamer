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
 * SECTION: element-tensor_repopush
 *
 * Push elemnt to handle tensor repo
 *
 * @file	tensor_repopush.c
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_repopush.h"

/**
 * @brief tensor repository
 */
extern GstTensorRepo _repo;


GST_DEBUG_CATEGORY_STATIC (gst_tensor_repopush_debug);
#define GST_CAT_DEFAULT gst_tensor_repopush_debug

/**
 * @brief tensor_repopush properties
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
 * @brief tensor_repopush sink template
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

static void gst_tensor_repopush_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_repopush_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_repopush_dispose (GObject * object);

static gboolean gst_tensor_repopush_start (GstBaseSink * sink);
static gboolean gst_tensor_repopush_stop (GstBaseSink * sink);
static gboolean gst_tensor_repopush_event (GstBaseSink * sink,
    GstEvent * event);
static gboolean gst_tensor_repopush_query (GstBaseSink * sink,
    GstQuery * query);
static GstFlowReturn gst_tensor_repopush_render (GstBaseSink * sink,
    GstBuffer * buffer);
static GstFlowReturn gst_tensor_repopush_render_list (GstBaseSink * sink,
    GstBufferList * buffer_list);
static gboolean gst_tensor_repopush_set_caps (GstBaseSink * sink,
    GstCaps * caps);
static GstCaps *gst_tensor_repopush_get_caps (GstBaseSink * sink,
    GstCaps * filter);


#define gst_tensor_repopush_parent_class parent_class
G_DEFINE_TYPE (GstTensorRepoPush, gst_tensor_repopush, GST_TYPE_BASE_SINK);

/**
 * @brief class initialization of tensor_repopush
 */
static void
gst_tensor_repopush_class_init (GstTensorRepoPushClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstBaseSinkClass *basesink_class;
  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);
  basesink_class = GST_BASE_SINK_CLASS (klass);

  gobject_class->set_property = gst_tensor_repopush_set_property;
  gobject_class->get_property = gst_tensor_repopush_get_property;
  gobject_class->dispose = gst_tensor_repopush_dispose;

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
      "TensorRepoPush",
      "Push/TensorRepo",
      "Push element to handle tensor repository",
      "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (element_class, &sink_template);

  basesink_class->start = GST_DEBUG_FUNCPTR (gst_tensor_repopush_start);
  basesink_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_repopush_stop);
  basesink_class->event = GST_DEBUG_FUNCPTR (gst_tensor_repopush_event);
  basesink_class->query = GST_DEBUG_FUNCPTR (gst_tensor_repopush_query);
  basesink_class->render = GST_DEBUG_FUNCPTR (gst_tensor_repopush_render);
  basesink_class->render_list =
      GST_DEBUG_FUNCPTR (gst_tensor_repopush_render_list);
  basesink_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_repopush_set_caps);
  basesink_class->get_caps = GST_DEBUG_FUNCPTR (gst_tensor_repopush_get_caps);
}

/**
 * @brief initialization of tensor_repopush
 */
static void
gst_tensor_repopush_init (GstTensorRepoPush * self)
{
  GstBaseSink *basesink;
  basesink = GST_BASE_SINK (self);

  gst_tensor_repo_init ();

  silent_debug ("GstTensorRepo is sucessfully initailzed");

  self->silent = DEFAULT_SILENT;
  self->signal_rate = DEFAULT_SIGNAL_RATE;
  self->last_render_time = GST_CLOCK_TIME_NONE;
  self->in_caps = NULL;

  gst_base_sink_set_qos_enabled (basesink, DEFAULT_QOS);
}

/**
 * @brief set property vmethod
 */
static void
gst_tensor_repopush_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorRepoPush *self;
  self = GST_TENSOR_REPOPUSH (object);

  switch (prop_id) {
    case PROP_SIGNAL_RATE:
      self->signal_rate = g_value_get_uint (value);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_SLOT:
      self->myid = g_value_get_uint (value);
      gst_tensor_repo_add_data (self->myid);
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
gst_tensor_repopush_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorRepoPush *self;
  self = GST_TENSOR_REPOPUSH (object);
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
gst_tensor_repopush_dispose (GObject * object)
{
  GstTensorRepoPush *self;
  self = GST_TENSOR_REPOPUSH (object);
  if (self->in_caps)
    gst_caps_unref (self->in_caps);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief start vmethod implementation
 */
static gboolean
gst_tensor_repopush_start (GstBaseSink * sink)
{
  return TRUE;
}

/**
 * @brief stop vmethod implementation
 */
static gboolean
gst_tensor_repopush_stop (GstBaseSink * sink)
{
  return TRUE;
}

/**
 * @brief Handle events.
 *
 * GstBaseSink method implementation.
 */
static gboolean
gst_tensor_repopush_event (GstBaseSink * sink, GstEvent * event)
{
  GstTensorRepoPush *self;
  GstEventType type;

  self = GST_TENSOR_REPOPUSH (sink);
  type = GST_EVENT_TYPE (event);

  silent_debug ("received event %s", GST_EVENT_TYPE_NAME (event));

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
gst_tensor_repopush_query (GstBaseSink * sink, GstQuery * query)
{
  GstQueryType type;
  GstFormat format;

  type = GST_QUERY_TYPE (query);

  switch (type) {
    case GST_QUERY_SEEKING:
      /** tensor sink does not support seeking */
      gst_query_parse_seeking (query, &format, NULL, NULL, NULL);
      gst_query_set_seeking (query, format, FALSE, 0, -1);
      return TRUE;

    default:
      break;
  }

  return GST_BASE_SINK_CLASS (parent_class)->query (sink, query);
}


/**
 * @brief push GstBuffer
 */
static void
gst_tensor_repopush_render_buffer (GstTensorRepoPush * self, GstBuffer * buffer)
{
  GstClockTime now = GST_CLOCK_TIME_NONE;
  guint signal_rate;
  gboolean notify = FALSE;
  g_return_if_fail (GST_IS_TENSOR_REPOPUSH (self));

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
    ret = gst_tensor_repo_push_buffer (self->myid, buffer);
    if (!ret)
      GST_ELEMENT_ERROR (self, RESOURCE, WRITE,
          ("Cannot push buffer into repo [key: %d]", self->myid), NULL);
  }
}

/**
 * @brief render vmethod implementation
 */
static GstFlowReturn
gst_tensor_repopush_render (GstBaseSink * sink, GstBuffer * buffer)
{
  GstTensorRepoPush *self;
  self = GST_TENSOR_REPOPUSH (sink);


  gst_tensor_repopush_render_buffer (self, buffer);
  return GST_FLOW_OK;
}

/**
 * @brief render list vmethod implementation
 */
static GstFlowReturn
gst_tensor_repopush_render_list (GstBaseSink * sink,
    GstBufferList * buffer_list)
{
  GstTensorRepoPush *self;
  GstBuffer *buffer;
  guint i;
  guint num_buffers;

  self = GST_TENSOR_REPOPUSH (sink);
  num_buffers = gst_buffer_list_length (buffer_list);

  for (i = 0; i < num_buffers; i++) {
    buffer = gst_buffer_list_get (buffer_list, i);
    gst_tensor_repopush_render_buffer (self, buffer);
  }

  return GST_FLOW_OK;
}

/**
 * @brief set_caps vmethod implementation
 */
static gboolean
gst_tensor_repopush_set_caps (GstBaseSink * sink, GstCaps * caps)
{
  GstTensorRepoPush *self;

  self = GST_TENSOR_REPOPUSH (sink);
  gst_caps_replace (&self->in_caps, caps);

  return TRUE;
}

/**
 * @brief get_caps vmethod implementation
 */
static GstCaps *
gst_tensor_repopush_get_caps (GstBaseSink * sink, GstCaps * filter)
{
  GstTensorRepoPush *self;
  GstCaps *caps;

  self = GST_TENSOR_REPOPUSH (sink);

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
NNSTREAMER_PLUGIN_INIT (tensor_repopush)
{
  GST_DEBUG_CATEGORY_INIT (gst_tensor_repopush_debug, "tensor_repopush",
      0, "tensor_repopush element");

  return gst_element_register (plugin, "tensor_repopush",
      GST_RANK_NONE, GST_TYPE_TENSOR_REPOPUSH);
}
