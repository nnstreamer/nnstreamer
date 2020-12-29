/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
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
#define silent_debug_caps(caps,msg) do { \
  if (DBG) { \
    if (caps) { \
      GstStructure *caps_s; \
      gchar *caps_s_string; \
      guint caps_size, caps_idx; \
      caps_size = gst_caps_get_size (caps);\
      for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
        caps_s = gst_caps_get_structure (caps, caps_idx); \
        caps_s_string = gst_structure_to_string (caps_s); \
        GST_DEBUG_OBJECT (self, msg " = %s", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

#define silent_debug_config(c,msg) do { \
  if (DBG) { \
    if (c) { \
      gchar *dim_str; \
      dim_str = gst_tensor_get_dimension_string ((c)->info.dimension); \
      GST_DEBUG_OBJECT (self, msg " type=%d dim=%s rate=%d/%d", (c)->info.type, dim_str, (c)->rate_n, (c)->rate_d); \
      g_free (dim_str); \
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
  PROP_CONCAT,
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
 * @brief The dimension index of frames in configured tensor.
 */
#define DEFAULT_FRAMES_DIMENSION (NNS_TENSOR_RANK_LIMIT - 1)

/**
 * @brief Flag to concatenate output buffer.
 */
#define DEFAULT_CONCAT TRUE

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

  GST_DEBUG_CATEGORY_INIT (gst_tensor_aggregator_debug, "tensor_aggregator", 0,
      "Element to aggregate tensor stream");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_aggregator_set_property;
  object_class->get_property = gst_tensor_aggregator_get_property;
  object_class->finalize = gst_tensor_aggregator_finalize;

  /**
   * GstTensorAggregator::frames-in:
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
   * GstTensorAggregator::frames-out:
   *
   * The number of frames in outgoing buffer. (buffer is a sinle tensor instance)
   * GstTensorAggregator calculates the size of outgoing frames and pushes a buffer to source pad.
   */
  g_object_class_install_property (object_class, PROP_FRAMES_OUT,
      g_param_spec_uint ("frames-out", "Frames in output",
          "The number of frames in outgoing buffer", 1, G_MAXUINT,
          DEFAULT_FRAMES_OUT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator::frames-flush:
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
   * GstTensorAggregator::frames-dim:
   *
   * The dimension index of frames in tensor.
   * If frames-in and frames-out are different, GstTensorAggregator has to change the dimension of tensor.
   * With this property, GstTensorAggregator changes the out-caps.
   */
  g_object_class_install_property (object_class, PROP_FRAMES_DIMENSION,
      g_param_spec_uint ("frames-dim", "Dimension index of frames",
          "The dimension index of frames in tensor",
          0, (NNS_TENSOR_RANK_LIMIT - 1), DEFAULT_FRAMES_DIMENSION,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator::concat:
   *
   * The flag to concatenate output buffer.
   * If concat is true and frames-out is larger than 1, GstTensorAggregator will concatenate the output buffer with the axis frames-dim.
   */
  g_object_class_install_property (object_class, PROP_CONCAT,
      g_param_spec_boolean ("concat", "Concat", "Concatenate output buffer",
          DEFAULT_CONCAT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorAggregator::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "TensorAggregator",
      "Filter/Tensor",
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
  self->concat = DEFAULT_CONCAT;

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
      self->frames_dim = g_value_get_uint (value);
      break;
    case PROP_CONCAT:
      self->concat = g_value_get_boolean (value);
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
      g_value_set_uint (value, self->frames_dim);
      break;
    case PROP_CONCAT:
      g_value_set_boolean (value, self->concat);
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

  GST_DEBUG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *in_caps;
      GstCaps *out_caps;

      gst_event_parse_caps (event, &in_caps);
      silent_debug_caps (in_caps, "in-caps");

      if (gst_tensor_aggregator_parse_caps (self, in_caps)) {
        gboolean ret = FALSE;

        out_caps = gst_tensor_caps_from_config (&self->out_config);
        silent_debug_caps (out_caps, "out-caps");

        ret = gst_pad_set_caps (self->srcpad, out_caps);

        gst_event_unref (event);
        gst_caps_unref (out_caps);

        return ret;
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

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
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
      gboolean res = FALSE;

      gst_query_parse_accept_caps (query, &caps);
      silent_debug_caps (caps, "accept-caps");

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
 * @brief This function handles src pad query.
 */
static gboolean
gst_tensor_aggregator_src_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstTensorAggregator *self;

  self = GST_TENSOR_AGGREGATOR (parent);

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
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
 * @brief Check tensor dimension and axis to concatenate data.
 * @param self this pointer to GstTensorAggregator
 * @param info tensor info for one frame
 * @return True if needed to concatenate
 */
static gboolean
gst_tensor_aggregator_check_concat_axis (GstTensorAggregator * self,
    const GstTensorInfo * info)
{
  guint i;

  g_assert (info != NULL); /** Internal error. Caller should've checked it */

  /**
   * Check condition to concatenate data.
   */
  if (self->concat && self->frames_out > 1) {
    for (i = self->frames_dim + 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
      if (info->dimension[i] > 1) {
        /** concatenate data */
        return TRUE;
      }
    }
  }

  return FALSE;
}

/**
 * @brief Change the data in buffer with given axis.
 * @param self this pointer to GstTensorAggregator
 * @param outbuf buffer to be concatenated
 * @param info tensor info for one frame
 */
static gboolean
gst_tensor_aggregator_concat (GstTensorAggregator * self, GstBuffer * outbuf,
    const GstTensorInfo * info)
{
  GstBuffer *srcbuf;
  GstMapInfo src_info, dest_info;
  guint f;
  gsize block_size;
  gsize src_idx, dest_idx;
  gsize frame_size;

  frame_size = gst_tensor_info_get_size (info);
  g_assert (frame_size > 0); /** Internal error */

  srcbuf = gst_buffer_copy (outbuf);
  outbuf = gst_buffer_make_writable (outbuf);

  if (FALSE == gst_buffer_map (srcbuf, &src_info, GST_MAP_READ)) {
    ml_logf ("Failed to map source buffer with tensor_aggregator.\n");
    gst_buffer_unref (srcbuf);
    return FALSE;
  }
  if (FALSE == gst_buffer_map (outbuf, &dest_info, GST_MAP_WRITE)) {
    ml_logf ("Failed to map destination buffer with tensor_aggregator.\n");
    gst_buffer_unmap (srcbuf, &src_info);
    gst_buffer_unref (srcbuf);
    return FALSE;
  }

  /**
   * Concatenate output buffer with given axis (frames-dim)
   * If frames-dim is equal to (NNS_TENSOR_RANK_LIMIT - 1), nothing to do.
   * (In this case, this function will not be called. See gst_tensor_aggregator_check_concat_axis ())
   ********************************************************************
   * Ex1) concatenate 2 frames with dimension 3:4:2:1
   ********************************************************************
   * frame 1
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124] ]
   *   ]
   *
   * frame 2
   *   [
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   ********************************************************************
   * 1-1. result with frames-dim 3 (3:4:2:2)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124] ]
   *   ]
   *   [
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   ********************************************************************
   * 1-2. result with frames-dim 2 (3:4:4:1)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124] ]
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   ********************************************************************
   * 1-3. result with frames-dim 1 (3:8:2:1)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112]
   *       [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124]
   *       [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   ********************************************************************
   * 1-4. result with frames-dim 0 (6:4:2:1)
   *   [
   *     [ [1101 1102 1103 2101 2102 2103] [1104 1105 1106 2104 2105 2106]
   *       [1107 1108 1109 2107 2108 2109] [1110 1111 1112 2110 2111 2112] ]
   *     [ [1113 1114 1115 2113 2114 2115] [1116 1117 1118 2116 2117 2118]
   *       [1119 1120 1121 2119 2120 2121] [1122 1123 1124 2122 2123 2124] ]
   *   ]
   ********************************************************************
   * Ex2) concatenate 2 frames with dimension 3:4:2:2
   ********************************************************************
   * frame 1
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124] ]
   *   ]
   *   [
   *     [ [1201 1202 1203] [1204 1205 1206] [1207 1208 1209] [1210 1211 1212] ]
   *     [ [1213 1214 1215] [1216 1217 1218] [1219 1220 1221] [1222 1223 1224] ]
   *   ]
   *
   * frame 2
   *   [
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   *   [
   *     [ [2201 2202 2203] [2204 2205 2206] [2207 2208 2209] [2210 2211 2212] ]
   *     [ [2213 2214 2215] [2216 2217 2218] [2219 2220 2221] [2222 2223 2224] ]
   *   ]
   ********************************************************************
   * 2-1. result with frames-dim 3 (3:4:2:4)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124] ]
   *   ]
   *   [
   *     [ [1201 1202 1203] [1204 1205 1206] [1207 1208 1209] [1210 1211 1212] ]
   *     [ [1213 1214 1215] [1216 1217 1218] [1219 1220 1221] [1222 1223 1224] ]
   *   ]
   *   [
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   *   [
   *     [ [2201 2202 2203] [2204 2205 2206] [2207 2208 2209] [2210 2211 2212] ]
   *     [ [2213 2214 2215] [2216 2217 2218] [2219 2220 2221] [2222 2223 2224] ]
   *   ]
   ********************************************************************
   * 2-2. result with frames-dim 2 (3:4:4:2)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124] ]
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   *   [
   *     [ [1201 1202 1203] [1204 1205 1206] [1207 1208 1209] [1210 1211 1212] ]
   *     [ [1213 1214 1215] [1216 1217 1218] [1219 1220 1221] [1222 1223 1224] ]
   *     [ [2201 2202 2203] [2204 2205 2206] [2207 2208 2209] [2210 2211 2212] ]
   *     [ [2213 2214 2215] [2216 2217 2218] [2219 2220 2221] [2222 2223 2224] ]
   *   ]
   ********************************************************************
   * 2-3. result with frames-dim 1 (3:8:2:2)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112]
   *       [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *     [ [1113 1114 1115] [1116 1117 1118] [1119 1120 1121] [1122 1123 1124]
   *       [2113 2114 2115] [2116 2117 2118] [2119 2120 2121] [2122 2123 2124] ]
   *   ]
   *   [
   *     [ [1201 1202 1203] [1204 1205 1206] [1207 1208 1209] [1210 1211 1212]
   *       [2201 2202 2203] [2204 2205 2206] [2207 2208 2209] [2210 2211 2212] ]
   *     [ [1213 1214 1215] [1216 1217 1218] [1219 1220 1221] [1222 1223 1224]
   *       [2213 2214 2215] [2216 2217 2218] [2219 2220 2221] [2222 2223 2224] ]
   *   ]
   ********************************************************************
   * 2-4. result with frames-dim 0 (6:4:2:2)
   *   [
   *     [ [1101 1102 1103 2101 2102 2103] [1104 1105 1106 2104 2105 2106]
   *       [1107 1108 1109 2107 2108 2109] [1110 1111 1112 2110 2111 2112] ]
   *     [ [1113 1114 1115 2113 2114 2115] [1116 1117 1118 2116 2117 2118]
   *       [1119 1120 1121 2119 2120 2121] [1122 1123 1124 2122 2123 2124] ]
   *   ]
   *   [
   *     [ [1201 1202 1203 2201 2202 2203] [1204 1205 1206 2204 2205 2206]
   *       [1207 1208 1209 2207 2208 2209] [1210 1211 1212 2210 2211 2212] ]
   *     [ [1213 1214 1215 2213 2214 2215] [1216 1217 1218 2216 2217 2218]
   *       [1219 1220 1221 2219 2220 2221] [1222 1223 1224 2222 2223 2224] ]
   *   ]
   ********************************************************************
   * Ex3) concatenate 2 frames with dimension 3:4:1:1
   ********************************************************************
   * frame 1
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *   ]
   *
   * frame 2
   *   [
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *   ]
   ********************************************************************
   * 3-1. result with frames-dim 3 (3:4:1:2)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *   ]
   *   [
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *   ]
   ********************************************************************
   * 3-2. result with frames-dim 2 (3:4:2:1)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112] ]
   *     [ [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *   ]
   ********************************************************************
   * 3-3. result with frames-dim 1 (3:8:1:1)
   *   [
   *     [ [1101 1102 1103] [1104 1105 1106] [1107 1108 1109] [1110 1111 1112]
   *       [2101 2102 2103] [2104 2105 2106] [2107 2108 2109] [2110 2111 2112] ]
   *   ]
   ********************************************************************
   * 3-4. result with frames-dim 0 (6:4:1:1)
   *   [
   *     [ [1101 1102 1103 2101 2102 2103] [1104 1105 1106 2104 2105 2106]
   *       [1107 1108 1109 2107 2108 2109] [1110 1111 1112 2110 2111 2112] ]
   *   ]
   ********************************************************************
   */

  /** get block size */
  block_size = gst_tensor_get_element_size (info->type);
  for (f = 0; f <= self->frames_dim; f++) {
    block_size *= info->dimension[f];
  }

  src_idx = dest_idx = 0;

  do {
    for (f = 0; f < self->frames_out; f++) {
      nns_memcpy (dest_info.data + dest_idx,
          src_info.data + src_idx + (frame_size * f), block_size);
      dest_idx += block_size;
    }

    src_idx += block_size;

    g_assert (src_idx <= frame_size);
    g_assert (dest_idx <= dest_info.size);
  } while (src_idx < frame_size);

  gst_buffer_unmap (srcbuf, &src_info);
  gst_buffer_unmap (outbuf, &dest_info);

  gst_buffer_unref (srcbuf);

  return TRUE;
}

/**
 * @brief Push the buffer to source pad. (Concatenate the buffer if needed)
 */
static GstFlowReturn
gst_tensor_aggregator_push (GstTensorAggregator * self, GstBuffer * outbuf,
    gsize frame_size)
{
  GstTensorInfo info;

  /** tensor info for one frame */
  info = self->out_config.info;
  g_assert (self->frames_dim < NNS_TENSOR_RANK_LIMIT);
  info.dimension[self->frames_dim] /= self->frames_out;

  if (frame_size != gst_tensor_info_get_size (&info) || frame_size == 0U) {
    ml_logf
        ("Invalid output capability of tensor_aggregator. Frame size = %"
        G_GSIZE_FORMAT "\n", frame_size);
    return GST_FLOW_ERROR;
  }

  if (gst_tensor_aggregator_check_concat_axis (self, &info)) {
    /** change data in buffer with given axis */
    if (FALSE == gst_tensor_aggregator_concat (self, outbuf, &info))
      return GST_FLOW_ERROR;
  }

  return gst_pad_push (self->srcpad, outbuf);
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
  guint frames_in, frames_out, frames_flush;
  GstClockTime duration;

  self = GST_TENSOR_AGGREGATOR (parent);
  g_assert (self->tensor_configured);

  buf_size = gst_buffer_get_size (buf);
  g_return_val_if_fail (buf_size > 0, GST_FLOW_ERROR);

  frames_in = self->frames_in;
  frames_out = self->frames_out;
  frames_flush = self->frames_flush;
  frame_size = buf_size / frames_in;

  if (frames_in == frames_out) {
    /** push the incoming buffer (do concat if needed) */
    return gst_tensor_aggregator_push (self, buf, frame_size);
  }

  adapter = self->adapter;
  g_assert (adapter != NULL);

  duration = GST_BUFFER_DURATION (buf);
  if (GST_CLOCK_TIME_IS_VALID (duration)) {
    /** supposed same duration for incoming buffer */
    duration = gst_util_uint64_scale_int (duration, frames_out, frames_in);
  }

  gst_adapter_push (adapter, buf);

  out_size = frame_size * frames_out;
  g_assert (out_size > 0);

  while ((avail = gst_adapter_available (adapter)) >= out_size &&
      ret == GST_FLOW_OK) {
    GstBuffer *outbuf;
    GstClockTime pts, dts;
    guint64 pts_dist, dts_dist;
    gsize flush;

    pts = gst_adapter_prev_pts (adapter, &pts_dist);
    dts = gst_adapter_prev_dts (adapter, &dts_dist);

    /**
     * Update timestamp.
     * If frames-in is larger then frames-out, the same timestamp (pts and dts) would be returned.
     */
    if (frames_in > 1) {
      gint fn, fd;

      fn = self->in_config.rate_n;
      fd = self->in_config.rate_d;

      if (fn > 0 && fd > 0) {
        if (GST_CLOCK_TIME_IS_VALID (pts)) {
          pts +=
              gst_util_uint64_scale_int (pts_dist * fd, GST_SECOND,
              fn * frame_size);
        }

        if (GST_CLOCK_TIME_IS_VALID (dts)) {
          dts +=
              gst_util_uint64_scale_int (dts_dist * fd, GST_SECOND,
              fn * frame_size);
        }
      }
    }

    outbuf = gst_adapter_get_buffer (adapter, out_size);
    outbuf = gst_buffer_make_writable (outbuf);

    /** set timestamp */
    GST_BUFFER_PTS (outbuf) = pts;
    GST_BUFFER_DTS (outbuf) = dts;
    GST_BUFFER_DURATION (outbuf) = duration;

    ret = gst_tensor_aggregator_push (self, outbuf, frame_size);

    /** flush data */
    if (frames_flush > 0) {
      flush = frame_size * frames_flush;

      if (flush > avail) {
        /**
         * @todo flush data
         * Invalid state, tried to flush large size.
         * We have to determine how to handle this case. (flush the out-size or all available bytes)
         * Now all available bytes in adapter will be flushed.
         */
        flush = avail;
      }
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
  GstTensorConfig config;

  /* tensor config info for given pad */
  if (pad == self->sinkpad) {
    config = self->in_config;
  } else {
    config = self->out_config;
  }

  /* caps from tensor config info */
  caps = gst_tensor_caps_from_config (&config);

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
  uint32_t per_frame;
  guint count;

  g_return_val_if_fail (caps != NULL, FALSE);
  g_return_val_if_fail (gst_caps_is_fixed (caps), FALSE);

  structure = gst_caps_get_structure (caps, 0);
  if (!gst_structure_has_name (structure, "other/tensor")) {
    GST_ERROR_OBJECT (self, "Invalid caps");
    return FALSE;
  }

  if (!gst_tensor_config_from_structure (&config, structure) ||
      !gst_tensor_config_validate (&config)) {
    GST_ERROR_OBJECT (self, "Cannot configure tensor info");
    return FALSE;
  }

  /**
   * @todo flush data
   * Check properties to detect invalid case.
   * Assertion when in=5 out=10 flush=20 or in=10 out=5 flush=20
   */
  count = (self->frames_out + self->frames_in - 1) / self->frames_in;
  g_assert (self->frames_in * count >= self->frames_flush);

  self->in_config = config;

  /**
   * update dimension in output tensor.
   * e.g, in-dimension 2:200:200:1
   * if frames_out=10 and frames_dim=3, then out-dimension is 2:200:200:10.
   * if frames_out=10 and frames_dim=2, then out-dimension is 2:200:2000:1.
   */
  g_assert (self->frames_dim < NNS_TENSOR_RANK_LIMIT);
  g_assert ((config.info.dimension[self->frames_dim] % self->frames_in) == 0);
  per_frame = config.info.dimension[self->frames_dim] / self->frames_in;

  config.info.dimension[self->frames_dim] = per_frame * self->frames_out;

  self->out_config = config;
  self->tensor_configured = TRUE;

  silent_debug_config (&self->in_config, "in-tensor");
  silent_debug_config (&self->out_config, "out-tensor");
  return TRUE;
}
