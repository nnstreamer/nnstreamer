/* GStreamer
 *
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
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/**
 * @file	gsttensor_trainsink.h
 * @date	11 October 2022
 * @brief	GStreamer plugin to train tensor data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	nnfw <nnfw@samsung.com>
 * @bug		No known bugs except for NYI items
 * 
 * ## Example launch line
 * |[
 * gst-launch-1.0 gst-launch-1.0 videotestsrc !
 *    video/x-raw, format=RGB, width=640, height=480 !
 *    tensor_converter ! tensor_trainsink
 * ]|
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gsttensor_trainsink.h"

/**
 * @brief Default caps string for sink pad.
 */
#define CAPS_STRING GST_TENSORS_CAP_MAKE ("{ static, flexible }")

static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief Default dump property value.
 */
#define DEFAULT_DUMP FALSE

/**
 * @brief tensor_trainsink properties.
 */
enum
{
  PROP_0,
  PROP_DUMP,
};

GST_DEBUG_CATEGORY_STATIC (gst_tensor_trainsink_debug);
#define GST_CAT_DEFAULT gst_tensor_trainsink_debug
#define gst_tensor_trainsink_parent_class parent_class
G_DEFINE_TYPE (GstTensorTrainSink, gst_tensor_trainsink, GST_TYPE_BASE_SINK);

static void gst_tensor_trainsink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_trainsink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_trainsink_finalize (GObject * object);

static GstStateChangeReturn gst_tensor_trainsink_change_state (GstElement *
    element, GstStateChange transition);
static GstFlowReturn gst_tensor_trainsink_render (GstBaseSink * bsink,
    GstBuffer * buffer);
static GstCaps *gst_tensor_trainsink_get_caps (GstBaseSink * bsink,
    GstCaps * filter);
static gboolean gst_tensor_trainsink_set_caps (GstBaseSink * bsink,
    GstCaps * caps);
static gboolean gst_tensor_trainsink_event (GstBaseSink * bsink,
    GstEvent * event);
static gboolean gst_tensor_trainsink_query (GstBaseSink * sink,
    GstQuery * query);

/**
 * @brief Initialize tensor_trainsink class.
 */
static void
gst_tensor_trainsink_class_init (GstTensorTrainSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbase_sink_class;

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, "tensor_trainsink", 0,
      "Tensor train sink plugin");

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  gstbase_sink_class = GST_BASE_SINK_CLASS (klass);

  gobject_class->set_property = gst_tensor_trainsink_set_property;
  gobject_class->get_property = gst_tensor_trainsink_get_property;
  gobject_class->finalize = gst_tensor_trainsink_finalize;

  g_object_class_install_property (gobject_class, PROP_DUMP,
      g_param_spec_boolean ("dump", "Dump", "Dump buffer",
          DEFAULT_DUMP,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_PLAYING |
          G_PARAM_STATIC_STRINGS));


  gst_element_class_set_static_metadata (gstelement_class,
      "TensorTrain Sink",
      "Sink/Tensor",
      "Train tensor data using NN Frameworks", "Samsung Electronics Co., Ltd.");

  gst_element_class_add_static_pad_template (gstelement_class, &sinktemplate);

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_trainsink_change_state);

  gstbase_sink_class->render = GST_DEBUG_FUNCPTR (gst_tensor_trainsink_render);
  gstbase_sink_class->get_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_trainsink_get_caps);
  gstbase_sink_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_trainsink_set_caps);
  gstbase_sink_class->event = GST_DEBUG_FUNCPTR (gst_tensor_trainsink_event);
  gstbase_sink_class->query = GST_DEBUG_FUNCPTR (gst_tensor_trainsink_query);
}

/**
 * @brief Initialize tensor_trainsink.
 */
static void
gst_tensor_trainsink_init (GstTensorTrainSink * sink)
{
  sink->dump = DEFAULT_DUMP;
}

/**
 * @brief finalize tensor_trainsink.
 */
static void
gst_tensor_trainsink_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_trainsink properties.
 */
static void
gst_tensor_trainsink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorTrainSink *sink = GST_TENSOR_TRAINSINK (object);

  switch (prop_id) {
    case PROP_DUMP:
      sink->dump = g_value_get_boolean (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter tensor_trainsink properties.
 */
static void
gst_tensor_trainsink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorTrainSink *sink;

  sink = GST_TENSOR_TRAINSINK (object);

  switch (prop_id) {
    case PROP_DUMP:
      g_value_set_boolean (value, sink->dump);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Called when a buffer should be presented or ouput.
 */
static GstFlowReturn
gst_tensor_trainsink_render (GstBaseSink * bsink, GstBuffer * buf)
{
  GstTensorTrainSink *sink = GST_TENSOR_TRAINSINK_CAST (bsink);
  GstMapInfo info;

  GST_OBJECT_LOCK (sink);

  gst_buffer_map (buf, &info, GST_MAP_READ);
  if (sink->dump) {
    gst_util_dump_mem (info.data, info.size);
  }

  gst_buffer_unmap (buf, &info);
  GST_OBJECT_UNLOCK (sink);

  return GST_FLOW_OK;
}

/**
 * @brief Get caps of tensor_trainsink.
 */
static GstCaps *
gst_tensor_trainsink_get_caps (GstBaseSink * bsink, GstCaps * filter)
{
  GstTensorTrainSink *sink = GST_TENSOR_TRAINSINK (bsink);
  GstCaps *caps = NULL;

  GST_OBJECT_LOCK (sink);
  caps = gst_pad_get_pad_template_caps (GST_BASE_SINK_PAD (bsink));
  GST_OBJECT_UNLOCK (sink);

  GST_INFO_OBJECT (sink, "Got caps %" GST_PTR_FORMAT, caps);

  if (caps && filter) {
    GstCaps *intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }

  GST_DEBUG_OBJECT (sink, "result get caps: %" GST_PTR_FORMAT, caps);

  return caps;
}

/**
 * @brief Set caps of tensor_trainsink.
 */
static gboolean
gst_tensor_trainsink_set_caps (GstBaseSink * bsink, GstCaps * caps)
{
  GstTensorTrainSink *sink;

  sink = GST_TENSOR_TRAINSINK (bsink);
  GST_INFO_OBJECT (sink, "set caps %" GST_PTR_FORMAT, caps);

  return TRUE;
}


/**
 * @brief Change state of tensor_trainsink.
 */
static GstStateChangeReturn
gst_tensor_trainsink_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstTensorTrainSink *sink = GST_TENSOR_TRAINSINK (element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      GST_INFO_OBJECT (sink, "NULL_TO_READY");
      break;

    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (sink, "READY_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (sink, "PAUSED_TO_PLAYING");
      break;

    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (sink, "PLAYING_TO_PAUSED");
      break;

    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (sink, "PAUSED_TO_READY");
      break;

    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (sink, "READY_TO_NULL");
      break;

    default:
      break;
  }
  return ret;
}

/**
 * @brief Receive Event on tensor_trainsink.
 */
static gboolean
gst_tensor_trainsink_event (GstBaseSink * bsink, GstEvent * event)
{
  GstTensorTrainSink *sink;
  sink = GST_TENSOR_TRAINSINK (bsink);

  GST_INFO_OBJECT (sink, "got event (%s)",
      gst_event_type_get_name (GST_EVENT_TYPE (event)));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      GST_INFO_OBJECT (sink, "get GST_EVENT_EOS event..state is %d",
          GST_STATE (sink));
      break;
    case GST_EVENT_FLUSH_START:
      GST_INFO_OBJECT (sink, "get GST_EVENT_FLUSH_START event");
      break;
    case GST_EVENT_FLUSH_STOP:
      GST_INFO_OBJECT (sink, "get GST_EVENT_FLUSH_STOP event");
      break;
    default:
      break;
  }

  return GST_BASE_SINK_CLASS (parent_class)->event (bsink, event);
}

/**
 * @brief Perform a GstQuery on tensor_trainsink.
 */
static gboolean
gst_tensor_trainsink_query (GstBaseSink * bsink, GstQuery * query)
{
  gboolean ret;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_SEEKING:{
      GstFormat fmt;

      /* we don't supporting seeking */
      gst_query_parse_seeking (query, &fmt, NULL, NULL, NULL);
      gst_query_set_seeking (query, fmt, FALSE, 0, -1);
      ret = TRUE;
      break;
    }
    default:
      ret = GST_BASE_SINK_CLASS (parent_class)->query (bsink, query);
      break;
  }

  return ret;
}
