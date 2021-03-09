/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsrc.c
 * @date    08 Mar 2021
 * @brief   Subscribe a MQTT topic and push incoming data to the GStreamer pipeline
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/base/gstbasesrc.h>

#include "mqttsrc.h"

#define gst_mqtt_src_parent_class parent_class
G_DEFINE_TYPE (GstMqttSrc, gst_mqtt_src, GST_TYPE_BASE_SRC);

GST_DEBUG_CATEGORY_STATIC (gst_mqtt_src_debug);
#define GST_CAT_DEFAULT gst_mqtt_src_debug

enum
{
  PROP_0,

  PROP_LAST
};


/** Function prototype declarations */
static void
gst_mqtt_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void
gst_mqtt_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_mqtt_src_class_finalize (GObject * object);

static GstStateChangeReturn
gst_mqtt_src_change_state (GstElement * element, GstStateChange transition);

static gboolean gst_mqtt_src_start (GstBaseSrc * basesrc);
static gboolean gst_mqtt_src_stop (GstBaseSrc * basesrc);
static gboolean gst_mqtt_src_event (GstBaseSrc * basesrc, GstEvent * event);
static gboolean gst_mqtt_src_set_caps (GstBaseSrc * basesrc, GstCaps * caps);
static GstCaps *gst_mqtt_src_get_caps (GstBaseSrc * basesrc, GstCaps * filter);
static GstCaps *gst_mqtt_src_fixate (GstBaseSrc * basesrc, GstCaps * caps);
static void
gst_mqtt_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end);
static gboolean gst_mqtt_src_get_size (GstBaseSrc * basesrc, guint64 * size);
static gboolean gst_mqtt_src_is_seekable (GstBaseSrc * basesrc);
static GstFlowReturn
gst_mqtt_src_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf);
static GstFlowReturn
gst_mqtt_src_alloc (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf);
static GstFlowReturn
gst_mqtt_src_fill (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer * buf);

/** Function defintions */
static void
gst_mqtt_src_init (GstMqttSrc * self)
{
  /** @todo */
}

static void
gst_mqtt_src_class_init (GstMqttSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, GST_MQTT_ELEM_NAME_SRC, 0,
      "MQTT src");

  gobject_class->set_property = gst_mqtt_src_set_property;
  gobject_class->get_property = gst_mqtt_src_get_property;
  gobject_class->finalize = gst_mqtt_src_class_finalize;

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_mqtt_src_change_state);
  gst_element_class_set_static_metadata (gstelement_class,
      "MQTT Source",
      "Source/MQTT",
      "Subscribe a MQTT topic and push incoming data to the GStreamer pipeline",
      "Wook Song <wook16.song@samsung.com>");

  gstbasesrc_class->start = GST_DEBUG_FUNCPTR (gst_mqtt_src_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_mqtt_src_stop);
  gstbasesrc_class->event = GST_DEBUG_FUNCPTR (gst_mqtt_src_event);
  gstbasesrc_class->set_caps = GST_DEBUG_FUNCPTR (gst_mqtt_src_set_caps);
  gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_caps);
  gstbasesrc_class->fixate = GST_DEBUG_FUNCPTR (gst_mqtt_src_fixate);
  gstbasesrc_class->get_times = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_times);
  gstbasesrc_class->get_size = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_size);
  gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR (gst_mqtt_src_is_seekable);
  gstbasesrc_class->create = GST_DEBUG_FUNCPTR (gst_mqtt_src_create);
  gstbasesrc_class->alloc = GST_DEBUG_FUNCPTR (gst_mqtt_src_alloc);
  gstbasesrc_class->fill = GST_DEBUG_FUNCPTR (gst_mqtt_src_fill);
}

static void
gst_mqtt_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_mqtt_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_mqtt_src_class_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static GstStateChangeReturn
gst_mqtt_src_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;

  return ret;
}

static gboolean
gst_mqtt_src_start (GstBaseSrc * basesrc)
{
  return TRUE;
}

static gboolean
gst_mqtt_src_stop (GstBaseSrc * basesrc)
{
  return TRUE;
}

static gboolean
gst_mqtt_src_event (GstBaseSrc * basesrc, GstEvent * event)
{
  return TRUE;
}

static gboolean
gst_mqtt_src_set_caps (GstBaseSrc * basesrc, GstCaps * caps)
{
  return TRUE;
}

static GstCaps *
gst_mqtt_src_get_caps (GstBaseSrc * basesrc, GstCaps * filter)
{
  GstCaps *caps = gst_caps_new_empty ();

  return caps;
}

static GstCaps *
gst_mqtt_src_fixate (GstBaseSrc * basesrc, GstCaps * caps)
{
  caps = gst_caps_make_writable (caps);
  caps = GST_BASE_SRC_CLASS (parent_class)->fixate (basesrc, caps);

  return caps;
}

static void
gst_mqtt_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
  return;
}

static gboolean
gst_mqtt_src_get_size (GstBaseSrc * basesrc, guint64 * size)
{
  return TRUE;
}

static gboolean
gst_mqtt_src_is_seekable (GstBaseSrc * basesrc)
{
  return TRUE;
}

static GstFlowReturn
gst_mqtt_src_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf)
{
  return GST_FLOW_OK;
}

static GstFlowReturn
gst_mqtt_src_alloc (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf)
{
  return GST_FLOW_OK;
}

static GstFlowReturn
gst_mqtt_src_fill (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer * buf)
{
  return GST_FLOW_OK;
}
