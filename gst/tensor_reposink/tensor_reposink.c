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
 * Sink elemnt to handle tensor repo
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

#include "tensor_reposink.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) \
    debug_print (DBG, __VA_ARGS__)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_reposink_debug);
#define GST_CAT_DEFAULT gst_tensor_reposink_debug

/**
 * @brief tensor repository
 */
GstTensorRepo _repo;

/**
 * @brief tensor_reposink signals
 */
enum
{
  SIGNAL_NEW_DATA,
  SIGNAL_STREAM_START,
  SIGNAL_EOS,
  LAST_SIGNAL
};

/**
 * @brief tensor_reposink properties
 */
enum
{
  PROP_SIGNAL_RATE,
  PROP_EMIT_SIGNAL,
  PROP_SILENT
};

#define DEFAULT_EMIT_SIGNAL TRUE
#define DEFAULT_SIGNAL_RATE 0
#define DEFAULT_SILENT TRUE
#define DEFAULT_QOS TRUE

/**
 * @brief tensor_reposink sink template
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

static guint _tensor_reposink_signals[LAST_SIGNAL] = { 0 };

static void gst_tensor_reposink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_reposink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_reposink_dispose (GObject * object);
static void gst_tensor_reposink_finalize (GObject * object);

/** GstBaseSink method implementation */
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
  gobject_class->finalize = gst_tensor_reposink_finalize;

  g_object_class_install_property (gobject_class, PROP_SIGNAL_RATE,
      g_param_spec_uint ("signal-rate", "Signal rate",
          "New data signals per second (0 for unlimited, max 500)", 0, 500,
          DEFAULT_SIGNAL_RATE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_EMIT_SIGNAL,
      g_param_spec_boolean ("emit-signal", "Emit signal",
          "Emit signal for new data, stream start, eos", DEFAULT_EMIT_SIGNAL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  _tensor_reposink_signals[SIGNAL_STREAM_START] =
      g_signal_new ("stream-start", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST, G_STRUCT_OFFSET (GstTensorRepoSinkClass, stream_start),
      NULL, NULL, NULL, G_TYPE_NONE, 0, G_TYPE_NONE);

  _tensor_reposink_signals[SIGNAL_EOS] =
      g_signal_new ("eos", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstTensorRepoSinkClass, eos), NULL, NULL, NULL,
      G_TYPE_NONE, 0, G_TYPE_NONE);

  gst_element_class_set_static_metadata (element_class,
      "TensorRepoSink",
      "Sink/TensorRepo",
      "Sink element to handle tensor repository",
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

  if (!_repo.initialized)
    gst_tensor_repo_init ();

  self->data.config = NULL;
  self->data.buffer = NULL;

  g_mutex_init (&self->data.lock);
  g_cond_init (&self->data.cond);

  g_mutex_lock (&_repo.repo_lock);
  _repo.tensorsdata = g_slist_append (_repo.tensorsdata, &self->data);

  self->myid = _repo.num_buffer;
  _repo.num_buffer++;
  g_mutex_unlock (&_repo.repo_lock);

  self->silent = DEFAULT_SILENT;
  self->emit_signal = DEFAULT_EMIT_SIGNAL;
  self->signal_rate = DEFAULT_SIGNAL_RATE;
  self->last_render_time = GST_CLOCK_TIME_NONE;
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
  /* NYI */
}

/**
 * @brief get property vmethod
 */
static void
gst_tensor_reposink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  /* NYI */
}

/**
 * @brief dispose vmethod implementation
 */
static void
gst_tensor_reposink_dispose (GObject * object)
{
  GstTensorRepoSink *self;
  GstTensorData *data;
  self = GST_TENSOR_REPOSINK (object);

  GST_TENSOR_REPO_LOCK (self->myid);
  data = gst_tensor_repo_get_tensor (self->myid);
  gst_object_unref (data->config);
  gst_object_unref (data->buffer);
  GST_TENSOR_REPO_UNLOCK (self->myid);

  g_mutex_clear (GST_TENSOR_REPO_GET_LOCK (self->myid));
  g_cond_clear (GST_TENSOR_REPO_GET_COND (self->myid));

  gst_tensor_repo_remove_data (self->myid);
}

/**
 * @brief finalize vmethod implementation
 */
static void
gst_tensor_reposink_finalize (GObject * object)
{
  GstTensorRepoSink *self;

  self = GST_TENSOR_REPOSINK (object);
  g_mutex_clear (GST_TENSOR_REPO_GET_LOCK (self->myid));
  g_cond_clear (GST_TENSOR_REPO_GET_COND (self->myid));

  G_OBJECT_CLASS (parent_class)->finalize (object);
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
 * @brief event vmethod implementation
 */
static gboolean
gst_tensor_reposink_event (GstBaseSink * sink, GstEvent * event)
{
  GstTensorRepoSink *self;
  GstEventType type;

  self = GST_TENSOR_REPOSINK (sink);
  type = GST_EVENT_TYPE (event);

  switch (type) {
    case GST_EVENT_STREAM_START:
      break;

    case GST_EVENT_EOS:
      break;

    default:
      break;
  }

  silent_debug ("received event %s", GST_EVENT_TYPE_NAME (event));

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

  silent_debug ("received query %s", GST_QUERY_TYPE_NAME (query));
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
 * @brief render vmethod implementation
 */
static GstFlowReturn
gst_tensor_reposink_render (GstBaseSink * sink, GstBuffer * buffer)
{
  return GST_FLOW_OK;
}

/**
 * @brief render list vmethod implementation
 */
static GstFlowReturn
gst_tensor_reposink_render_list (GstBaseSink * sink,
    GstBufferList * buffer_list)
{
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

  GST_TENSOR_REPO_LOCK (self->myid);
  gst_caps_replace (&self->in_caps, caps);
  GST_TENSOR_REPO_UNLOCK (self->myid);

  if (DBG) {
    guint caps_size, i;

    caps_size = gst_caps_get_size (caps);
    silent_debug ("set caps, size is %d", caps_size);

    for (i = 0; i < caps_size; i++) {
      GstStructure *structure = gst_caps_get_structure (caps, i);
      gchar *str = gst_structure_to_string (structure);

      silent_debug ("[%d] %s", i, str);
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

  GST_TENSOR_REPO_LOCK (self->myid);
  caps = self->in_caps;

  if (caps) {
    if (filter) {
      caps = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
    } else {
      gst_caps_ref (caps);
    }
  }
  GST_TENSOR_REPO_UNLOCK (self->myid);

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

#ifndef SINGLE_BINARY
/**
 * @brief Definition for identifying tensor_sink plugin.
 *
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

/**
 * @brief Macro to define the entry point of the plugin.
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_reposink,
    "Sink element to handle tensor repository",
    gst_tensor_reposink_plugin_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer");
#endif
