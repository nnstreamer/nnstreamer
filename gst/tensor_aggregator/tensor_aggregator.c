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
 * SECTION:element-tensor_aggregator
 *
 * @file	tensor_aggregator.c
 * @date	29 August 2018
 * @brief	GStreamer plugin to aggregate tensor stream
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_aggregator.h"

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

#define silent_debug_caps(caps,msg) do {\
  if (DBG) { \
    if (caps) { \
      GstStructure *caps_s; \
      gchar *caps_s_string; \
      guint caps_size, caps_idx; \
      caps_size = gst_caps_get_size (caps);\
      for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
        caps_s = gst_caps_get_structure (caps, caps_idx); \
        caps_s_string = gst_structure_to_string (caps_s); \
        debug_print (TRUE, msg " = %s\n", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_aggregator_debug);
#define GST_CAT_DEFAULT gst_tensor_aggregator_debug

/**
 * @brief tensor_aggregator properties
 */
enum
{
  PROP_0,
  PROP_FRAMES_IN,
  PROP_FRAMES_OUT,
  PROP_FRAMES_FLUSH,
  PROP_FRAMES_DIMENSION,
  PROP_SILENT
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief The number of frames in input buffer.
 */
#define DEFAULT_FRAMES_IN 1

/**
 * @brief The number of frames in output buffer.
 */
#define DEFAULT_FRAMES_OUT 1

/**
 * @brief The number of frames to flush.
 */
#define DEFAULT_FRAMES_FLUSH 0

/**
 * @brief The index of frames in tensor dimension.
 */
#define DEFAULT_FRAMES_DIMENSION (-1)

/**
 * @brief Template for sink pad.
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

#define gst_tensor_aggregator_parent_class parent_class
G_DEFINE_TYPE (GstTensorAggregator, gst_tensor_aggregator, GST_TYPE_ELEMENT);

static void gst_tensor_aggregator_finalize (GObject * object);
static void gst_tensor_aggregator_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_aggregator_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

static gboolean gst_tensor_aggregator_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_aggregator_sink_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static gboolean gst_tensor_aggregator_src_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static GstFlowReturn gst_tensor_aggregator_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buf);
static GstStateChangeReturn
gst_tensor_aggregator_change_state (GstElement * element,
    GstStateChange transition);

static void gst_tensor_aggregator_reset (GstTensorAggregator * self);
static GstCaps *gst_tensor_aggregator_query_caps (GstTensorAggregator * self,
    GstPad * pad, GstCaps * filter);
static gboolean gst_tensor_aggregator_parse_caps (GstTensorAggregator * self,
    const GstCaps * caps);

/**
 * @brief Initialize the tensor_aggregator's class.
 */
static void
gst_tensor_aggregator_class_init (GstTensorAggregatorClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_aggregator_set_property;
  object_class->get_property = gst_tensor_aggregator_get_property;
  object_class->finalize = gst_tensor_aggregator_finalize;

  /**
   * GstTensorAggregator:frames-in:
   *
   * The number of frames in incoming buffer.
   * GstTensorAggregator itself cannot get the frames in buffer. (buffer is a sinle tensor instance)
   * GstTensorAggregator calculates the size of single frame with this property.
   */
  g_object_class_install_property (object_class, PROP_FRAMES_IN,
      g_param_spec_uint ("frames-in", "Frames in input",
          "The number of frames in incoming buffer", 1, G_MAXUINT,
          DEFAULT_FRAMES_IN, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator:frames-out:
   *
   * The number of frames in outgoing buffer. (buffer is a sinle tensor instance)
   * GstTensorAggregator calculates the size of outgoing frames and pushes a buffer to source pad.
   */
  g_object_class_install_property (object_class, PROP_FRAMES_OUT,
      g_param_spec_uint ("frames-out", "Frames in output",
          "The number of frames in outgoing buffer", 1, G_MAXUINT,
          DEFAULT_FRAMES_OUT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator:frames-flush:
   *
   * The number of frames to flush.
   * GstTensorAggregator flushes the bytes (N frames) in GstAdapter after pushing a buffer.
   * If set 0 (default value), all outgoing frames will be flushed.
   */
  g_object_class_install_property (object_class, PROP_FRAMES_FLUSH,
      g_param_spec_uint ("frames-flush", "Frames to flush",
          "The number of frames to flush (0 to flush all output)", 0, G_MAXUINT,
          DEFAULT_FRAMES_FLUSH, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator:frames-dim:
   *
   * The dimension index of frames in tensor.
   * If frames-in and frames-out are different, GstTensorAggregator has to change the dimension of tensor.
   * With this property, GstTensorAggregator changes the out-caps.
   * If set -1 (default value), GstTensorAggregator does not change the dimension in outgoing tensor.
   * (This may cause an error if in/out frames are different.)
   */
  g_object_class_install_property (object_class, PROP_FRAMES_DIMENSION,
      g_param_spec_int ("frames-dim", "Dimension index of frames",
          "The dimension index of frames in tensor",
          -1, (NNS_TENSOR_RANK_LIMIT - 1), DEFAULT_FRAMES_DIMENSION,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator:silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "TensorAggregator",
      "Converter/Tensor",
      "Element to aggregate tensor stream", "Samsung Electronics Co., Ltd.");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_template));

  element_class->change_state = gst_tensor_aggregator_change_state;
}

/**
 * @brief Initialize tensor_aggregator element.
 */
static void
gst_tensor_aggregator_init (GstTensorAggregator * self)
{
  /** setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_aggregator_sink_event));
  gst_pad_set_query_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_aggregator_sink_query));
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_aggregator_chain));
  GST_PAD_SET_PROXY_CAPS (self->sinkpad);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /** setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_query_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_aggregator_src_query));
  GST_PAD_SET_PROXY_CAPS (self->srcpad);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /** init properties */
  self->silent = DEFAULT_SILENT;
  self->frames_in = DEFAULT_FRAMES_IN;
  self->frames_out = DEFAULT_FRAMES_OUT;
  self->frames_flush = DEFAULT_FRAMES_FLUSH;
  self->frames_dim = DEFAULT_FRAMES_DIMENSION;

  self->adapter = gst_adapter_new ();
  gst_tensor_aggregator_reset (self);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_aggregator_finalize (GObject * object)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (object);

  gst_tensor_aggregator_reset (self);

  if (self->adapter) {
    g_object_unref (self->adapter);
    self->adapter = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_aggregator properties.
 */
static void
gst_tensor_aggregator_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (object);

  switch (prop_id) {
    case PROP_FRAMES_IN:
      self->frames_in = g_value_get_uint (value);
      break;
    case PROP_FRAMES_OUT:
      self->frames_out = g_value_get_uint (value);
      break;
    case PROP_FRAMES_FLUSH:
      self->frames_flush = g_value_get_uint (value);
      break;
    case PROP_FRAMES_DIMENSION:
      self->frames_dim = g_value_get_int (value);
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
 * @brief Getter for tensor_aggregator properties.
 */
static void
gst_tensor_aggregator_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (object);

  switch (prop_id) {
    case PROP_FRAMES_IN:
      g_value_set_uint (value, self->frames_in);
      break;
    case PROP_FRAMES_OUT:
      g_value_set_uint (value, self->frames_out);
      break;
    case PROP_FRAMES_FLUSH:
      g_value_set_uint (value, self->frames_flush);
      break;
    case PROP_FRAMES_DIMENSION:
      g_value_set_int (value, self->frames_dim);
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
 * @brief This function handles sink events.
 */
static gboolean
gst_tensor_aggregator_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (parent);

  GST_LOG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *in_caps;
      GstCaps *out_caps;

      silent_debug ("EVENT_CAPS");

      gst_event_parse_caps (event, &in_caps);
      silent_debug_caps (in_caps, "in-caps");

      if (gst_tensor_aggregator_parse_caps (self, in_caps)) {
        out_caps = gst_tensor_caps_from_config (&self->out_config);
        silent_debug_caps (out_caps, "out-caps");

        gst_pad_set_caps (self->srcpad, out_caps);
        gst_pad_push_event (self->srcpad, gst_event_new_caps (out_caps));
        gst_caps_unref (out_caps);
        return TRUE;
      }
      break;
    }
    case GST_EVENT_FLUSH_STOP:
      gst_tensor_aggregator_reset (self);
      break;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief This function handles sink pad query.
 */
static gboolean
gst_tensor_aggregator_sink_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (parent);

  GST_LOG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      caps = gst_tensor_aggregator_query_caps (self, pad, filter);

      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      return TRUE;
    }
    case GST_QUERY_ACCEPT_CAPS:
    {
      GstCaps *caps;
      GstCaps *template_caps;
      gboolean res;

      gst_query_parse_accept_caps (query, &caps);
      silent_debug_caps (caps, "accept-caps");

      template_caps = gst_pad_get_pad_template_caps (pad);
      res = gst_caps_can_intersect (template_caps, caps);

      gst_query_set_accept_caps_result (query, res);
      gst_caps_unref (template_caps);
      return TRUE;
    }
    default:
      break;
  }

  return gst_pad_query_default (pad, parent, query);
}

/**
 * @brief This function handles src pad query.
 */
static gboolean
gst_tensor_aggregator_src_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (parent);

  GST_LOG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      caps = gst_tensor_aggregator_query_caps (self, pad, filter);

      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
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
gst_tensor_aggregator_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstTensorAggregator *self;
  GstFlowReturn ret = GST_FLOW_OK;
  GstAdapter *adapter;
  gsize avail, buf_size, frame_size, out_size;
  guint frames_in, frames_out;

  self = GST_TENSOR_AGGREGATOR (parent);
  g_assert (self->tensor_configured);

  buf_size = gst_buffer_get_size (buf);
  g_return_val_if_fail (buf_size > 0, GST_FLOW_ERROR);

  frames_in = self->frames_in;
  frames_out = self->frames_out;

  if (frames_in == frames_out) {
    /** do nothing, push the incoming buffer  */
    return gst_pad_push (self->srcpad, buf);
  }

  adapter = self->adapter;
  g_assert (adapter != NULL);

  gst_adapter_push (adapter, buf);

  frame_size = buf_size / frames_in;
  out_size = frame_size * frames_out;
  g_assert (out_size > 0);

  while ((avail = gst_adapter_available (adapter)) >= out_size &&
      ret == GST_FLOW_OK) {
    GstBuffer *outbuf;
    GstClockTime pts, dts;
    gsize flush, offset;

    /** offset for last frame */
    offset = frame_size * (frames_out - 1);

    pts = gst_adapter_prev_pts_at_offset (adapter, offset, NULL);
    dts = gst_adapter_prev_dts_at_offset (adapter, offset, NULL);

    outbuf = gst_adapter_get_buffer (adapter, out_size);

    GST_BUFFER_PTS (outbuf) = pts;
    GST_BUFFER_DTS (outbuf) = dts;

    ret = gst_pad_push (self->srcpad, outbuf);

    /** flush data */
    if (self->frames_flush > 0) {
      flush = frame_size * self->frames_flush;
    } else {
      flush = out_size;
    }

    gst_adapter_flush (adapter, flush);
  }

  return ret;
}

/**
 * @brief Called to perform state change.
 */
static GstStateChangeReturn
gst_tensor_aggregator_change_state (GstElement * element,
    GstStateChange transition)
{
  GstTensorAggregator *self;
  GstStateChangeReturn ret;

  self = GST_TENSOR_AGGREGATOR (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_tensor_aggregator_reset (self);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_tensor_aggregator_reset (self);
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Clear and reset data.
 */
static void
gst_tensor_aggregator_reset (GstTensorAggregator * self)
{
  if (self->adapter) {
    gst_adapter_clear (self->adapter);
  }

  self->tensor_configured = FALSE;
  gst_tensor_config_init (&self->in_config);
  gst_tensor_config_init (&self->out_config);
}

/**
 * @brief Get pad caps for caps negotiation.
 */
static GstCaps *
gst_tensor_aggregator_query_caps (GstTensorAggregator * self, GstPad * pad,
    GstCaps * filter)
{
  GstCaps *caps;

  if (pad == self->sinkpad) {
    caps = gst_tensor_caps_from_config (&self->in_config);
  } else {
    caps = gst_tensor_caps_from_config (&self->out_config);
  }

  silent_debug_caps (caps, "caps");
  silent_debug_caps (filter, "filter");

  if (caps && filter) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }

  return caps;
}

/**
 * @brief Parse caps and set tensor info.
 */
static gboolean
gst_tensor_aggregator_parse_caps (GstTensorAggregator * self,
    const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorConfig config;

  g_return_val_if_fail (caps != NULL, FALSE);
  g_return_val_if_fail (gst_caps_is_fixed (caps), FALSE);

  structure = gst_caps_get_structure (caps, 0);
  if (!gst_structure_has_name (structure, "other/tensor")) {
    silent_debug ("invalid caps");
    return FALSE;
  }

  if (!gst_tensor_config_from_structure (&config, structure) ||
      !gst_tensor_config_validate (&config)) {
    silent_debug ("cannot configure tensor info");
    return FALSE;
  }

  self->in_config = config;

  /** update dimension in output tensor */
  if (self->frames_dim >= 0) {
    config.dimension[self->frames_dim] = self->frames_out;
  }

  self->out_config = config;
  self->tensor_configured = TRUE;
  return TRUE;
}

/**
 * @brief Function to initialize the plugin.
 *
 * See GstPluginInitFunc() for more details.
 */
static gboolean
gst_tensor_aggregator_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_tensor_aggregator_debug, "tensor_aggregator",
      0, "tensor_aggregator element");

  return gst_element_register (plugin, "tensor_aggregator",
      GST_RANK_NONE, GST_TYPE_TENSOR_AGGREGATOR);
}

/**
 * @brief Definition for identifying tensor_aggregator plugin.
 *
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_aggregator"
#endif

/**
 * @brief Macro to define the entry point of the plugin.
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_aggregator,
    "GStreamer plugin to aggregate tensor stream",
    gst_tensor_aggregator_plugin_init, VERSION, "LGPL", "GStreamer",
    "http://gstreamer.net/");
