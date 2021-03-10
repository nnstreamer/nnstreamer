/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsink.c
 * @date    08 Mar 2021
 * @brief   Publish incoming data streams as a MQTT topic
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/base/gstbasesink.h>

#include "mqttsink.h"

static GstStaticPadTemplate sink_pad_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

#define gst_mqtt_sink_parent_class parent_class
G_DEFINE_TYPE (GstMqttSink, gst_mqtt_sink, GST_TYPE_BASE_SINK);

GST_DEBUG_CATEGORY_STATIC (gst_mqtt_sink_debug);
#define GST_CAT_DEFAULT gst_mqtt_sink_debug

enum
{
  PROP_0,

  PROP_NUM_BUFFERS,

  PROP_LAST
};

enum
{
  DEFAULT_NUM_BUFFERS = -1,
  DEFAULT_QOS = TRUE,
  DEFAULT_SYNC = FALSE,
};

/** Function prototype declarations */
static void
gst_mqtt_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void
gst_mqtt_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_mqtt_sink_class_finalize (GObject * object);

static GstStateChangeReturn
gst_mqtt_sink_change_state (GstElement * element, GstStateChange transition);

static gboolean gst_mqtt_sink_start (GstBaseSink * basesink);
static gboolean gst_mqtt_sink_stop (GstBaseSink * basesink);
static gboolean gst_mqtt_sink_query (GstBaseSink * basesink, GstQuery * query);
static GstFlowReturn
gst_mqtt_sink_render (GstBaseSink * basesink, GstBuffer * buffer);
static GstFlowReturn
gst_mqtt_sink_render_list (GstBaseSink * basesink, GstBufferList * list);
static gboolean gst_mqtt_sink_event (GstBaseSink * basesink, GstEvent * event);
static gint gst_mqtt_sink_get_num_buffers (GstMqttSink * self);
static void gst_mqtt_sink_set_num_buffers (GstMqttSink * self, gint num);

/**
 * @brief Initialize GstMqttSink object
 */
static void
gst_mqtt_sink_init (GstMqttSink * self)
{
  GstBaseSink *basesink = GST_BASE_SINK (self);

  /** init mqttsink properties */
  self->num_buffers = DEFAULT_NUM_BUFFERS;

  /** init basesink properties */
  gst_base_sink_set_qos_enabled (basesink, DEFAULT_QOS);
  gst_base_sink_set_sync (basesink, DEFAULT_SYNC);
}

/**
 * @brief Initialize GstMqttSinkClass object
 */
static void
gst_mqtt_sink_class_init (GstMqttSinkClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, GST_MQTT_ELEM_NAME_SINK, 0,
      "MQTT sink");

  gobject_class->set_property = gst_mqtt_sink_set_property;
  gobject_class->get_property = gst_mqtt_sink_get_property;
  gobject_class->finalize = gst_mqtt_sink_class_finalize;

  g_object_class_install_property (gobject_class, PROP_NUM_BUFFERS,
      g_param_spec_int ("num-buffers", "num-buffers",
          "Number of (remaining) buffers to accept until sending EOS event (-1 = no limit)",
          -1, G_MAXINT, DEFAULT_NUM_BUFFERS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gstelement_class->change_state = gst_mqtt_sink_change_state;

  gstbasesink_class->start = GST_DEBUG_FUNCPTR (gst_mqtt_sink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR (gst_mqtt_sink_stop);
  gstbasesink_class->query = GST_DEBUG_FUNCPTR (gst_mqtt_sink_query);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR (gst_mqtt_sink_render);
  gstbasesink_class->render_list =
      GST_DEBUG_FUNCPTR (gst_mqtt_sink_render_list);
  gstbasesink_class->event = GST_DEBUG_FUNCPTR (gst_mqtt_sink_event);

  gst_element_class_set_static_metadata (gstelement_class,
      "MQTT sink", "Sink/MQTT",
      "Publish incoming data streams as a MQTT topic",
      "Wook Song <wook16.song@samsung.com>");
  gst_element_class_add_static_pad_template (gstelement_class,
      &sink_pad_template);
}

/**
 * @brief The setter for the mqttsink's properties
 */
static void
gst_mqtt_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMqttSink *self = GST_MQTT_SINK (object);

  switch (prop_id) {
    case PROP_NUM_BUFFERS:
      gst_mqtt_sink_set_num_buffers (self, g_value_get_int (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief The getter for the mqttsink's properties
 */
static void
gst_mqtt_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstMqttSink *self = GST_MQTT_SINK (object);

  switch (prop_id) {
    case PROP_NUM_BUFFERS:
      g_value_set_int (value, gst_mqtt_sink_get_num_buffers (self));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Finalize GstMqttSinkClass object
 */
static void
gst_mqtt_sink_class_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Handle mqttsink's state change
 */
static GstStateChangeReturn
gst_mqtt_sink_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstMqttSink *mqttsink = GST_MQTT_SINK (element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      GST_INFO_OBJECT (mqttsink, "GST_STATE_CHANGE_NULL_TO_READY");
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (mqttsink, "GST_STATE_CHANGE_READY_TO_PAUSED");
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (mqttsink, "GST_STATE_CHANGE_PAUSED_TO_PLAYING");
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (mqttsink, "GST_STATE_CHANGE_PLAYING_TO_PAUSED");
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (mqttsink, "GST_STATE_CHANGE_PAUSED_TO_READY");
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (mqttsink, "GST_STATE_CHANGE_READY_TO_NULL");
    default:
      break;
  }

  return ret;
}

/**
 * @brief Start mqttsink, called when state changed null to ready
 */
static gboolean
gst_mqtt_sink_start (GstBaseSink * basesink)
{
  return TRUE;
}

/**
 * @brief Stop mqttsink, called when state changed ready to null
 */
static gboolean
gst_mqtt_sink_stop (GstBaseSink * basesink)
{
  return TRUE;
}

/**
 * @brief Perform queries on the element
 */
static gboolean
gst_mqtt_sink_query (GstBaseSink * basesink, GstQuery * query)
{
  gboolean ret = FALSE;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_SEEKING:{
      GstFormat fmt;

      /* GST_QUERY_SEEKING is not supported */
      gst_query_parse_seeking (query, &fmt, NULL, NULL, NULL);
      gst_query_set_seeking (query, fmt, FALSE, 0, -1);
      ret = TRUE;
      break;
    }
    default:{
      ret = GST_BASE_SINK_CLASS (parent_class)->query (basesink, query);
      break;
    }
  }

  return ret;
}

/**
 * @brief The callback to process each buffer receiving on the sink pad
 */
static GstFlowReturn
gst_mqtt_sink_render (GstBaseSink * basesink, GstBuffer * buffer)
{
  GstMqttSink *mqttsink = GST_MQTT_SINK (basesink);

  GST_OBJECT_LOCK (mqttsink);
  if (mqttsink->num_buffers == 0)
    goto ret_eos;

  if (mqttsink->num_buffers != -1)
    mqttsink->num_buffers -= 1;
  GST_OBJECT_UNLOCK (mqttsink);

  return GST_FLOW_OK;
ret_eos:
  {
    GST_OBJECT_UNLOCK (mqttsink);
    return GST_FLOW_EOS;
  }
}

/**
 * @brief The callback to process GstBufferList (instead of a single buffer)
 *        on the sink pad
 */
static GstFlowReturn
gst_mqtt_sink_render_list (GstBaseSink * basesink, GstBufferList * list)
{
  guint num_buffers = gst_buffer_list_length (list);
  GstFlowReturn ret;
  GstBuffer *buffer;
  guint i;

  for (i = 0; i < num_buffers; ++i) {
    buffer = gst_buffer_list_get (list, i);
    ret = gst_mqtt_sink_render (basesink, buffer);
    if (ret != GST_FLOW_OK)
      break;
  }

  return ret;
}

/**
 * @brief Handle events arriving on the sink pad
 */
static gboolean
gst_mqtt_sink_event (GstBaseSink * basesink, GstEvent * event)
{
  GstEventType type = GST_EVENT_TYPE (event);
  gboolean ret = FALSE;

  switch (type) {
    default:
      ret = GST_BASE_SINK_CLASS (parent_class)->event (basesink, event);
      break;
  }

  return ret;
}

/**
 * @brief Getter for the 'num-buffers' property.
 */
static gint
gst_mqtt_sink_get_num_buffers (GstMqttSink * self)
{
  gint num_buffers;

  GST_OBJECT_LOCK (self);
  num_buffers = self->num_buffers;
  GST_OBJECT_UNLOCK (self);

  return num_buffers;
}

/**
 * @brief Setter for the 'num-buffers' property
 */
static void
gst_mqtt_sink_set_num_buffers (GstMqttSink * self, gint num)
{
  GST_OBJECT_LOCK (self);
  self->num_buffers = num;
  GST_OBJECT_UNLOCK (self);
}
