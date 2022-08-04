/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_sink.h
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

enum
{
  PROP_0,

  /** @todo define props */

  PROP_LAST
};

#define gst_edgesink_parent_class parent_class
G_DEFINE_TYPE (GstEdgeSink, gst_edgesink, GST_TYPE_BASE_SINK);

static void gst_edgesink_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSepc * pspec);

static void gst_edgesink_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

static void gst_edgesink_finalize (GObject * object);

static gboolean gst_edgesink_start (GstBaseSink * basesink);
static GstFlowReturn gst_edgesink_render (GstBaseSink * basesink,
    GstBuffer * buf);
static gboolean gst_edgesink_set_caps (GstBaseSink * basesink, GstCaps * caps);

/**
 * @brief initialize the class
 */
static void
gst_edgesink_class_init (GstEdgeSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gselement_class;
  GstBaseSinkClass *gstbasesink_class;

  gstbasesink_class = (GstBaseSinkClass *) klass;
  gstelement_class = (GstElementClass *) gstbasesink_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_edgesink_set_property;
  gobject_class->get_property = gst_edgesink_get_property;
  gobject_class->finalize = gst_edgesink_finalize;

  /** @todo set props */

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
gst_edgesink_init (GstEdgeSink * sink)
{
  /** @todo set default value of props */
}


/**
 * @brief finalize the object
 */
static void
gst_edgesink_finalize (GObject * object)
{
  GstEdgeSink *self = GST_EDGESINK (object);
  /** @todo finalize - free all pointer in element */
  G_OBJECT_CLASS (parent_class)->finalize (object);
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
    /** @todo set prop */
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
    GParmSpec * pspec)
{
  GstEdgeSInk *self = GST_EDGESINK (object);

  switch (prop_id) {
    /** @todo props */
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief start processing of edgesink
 */
static gboolean
gst_edgesink_start (GstBaseSink * basesink)
{
  GstEdgeSink *sink = GST_EDGESINK (basesink);

  /** @todo start */
}

/**
 * @brief render buffer, send buffer
 */
static GstFlowReturn
gst_edgesink_render (GstBaseSink * basesink, GstBuffer * buf)
{
  /** @todo render, send data */
}

/**
 * @brief An implementation of the set_caps vmethod in GstBaseSinkClass
 */
static gboolean
gst_edgesink_set_caps (GstBaseSink * basesink, GstCaps * caps)
{
  /** @todo set caps */
}
