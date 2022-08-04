/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_src.c
 * @date    02 Aug 2022
 * @brief   Subscribe and push incoming data to the GStreamer pipeline
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "edge_src.h"

GST_DEBUG_CATEGORY_STATIC (gst_edgesrc_debug);
#define GST_CAT_DEFAULT gst_edgesrc_debug

/**
 * @brief the capabilities of the outputs 
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

/**
 * @brief edgesrc properties
 */
enum
{
  PROP_0,

  /** @todo define props */

  PROP_LAST
}

#define gst_edgesrc_parent_class parent_class
G_DEFINE_TYPE (GstEdgeSrc, gst_edgesrc, GST_TYPE_PUSH_SRC);

static void gst_edgesrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_edgesrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_edgesrc_class_finalize (GObject * object);

static GstStateChangeReturn gst_edgesrc_change_state (GstElement * element,
    GstStateChange transition);

static gboolean gst_edgesrc_start (GstBaseSrc * basesrc);
static gboolean gst_edgesrc_stop (GstBaseSrc * basesrc);
static GstCaps *gst_edgesrc_get_caps (GstBaseSrc * basesrc, GstCaps * filter);
static void gst_edgesrc_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end);
static gboolean gst_edgesrc_is_seekable (GstBaseSrc * basesrc);
static GstFlowReturn gst_edgesrc_create (GstBaseSrc * basesrc, guint64 offset,
    guint size, GstBuffer ** buf);
static gboolean gst_edgesrc_query (GstBaseSrc * basesrc, GstQuery * query);

/**
 * @brief initialize the class
 */
static void
gst_edgesrc_class_init (GstEdgeSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);

  gobject_class->set_property = gst_edgesrc_set_property;
  gobject_class->get_property = gst_edgesrc_get_property;
  gobject_class->finalize = gst_edgesrc_class_finalize;

  /** @todo set props */

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gelement_class,
      "EdgeSrc", "Source/Edge",
      "Subscribe and push incoming streams", "Samsung Electronics Co., Ltd.");

  gstelement_class->change_state = gst_edgesrc_change_state;

  gstbasesrc_class->start = gst_edgesrc_start;
  gstbasesrc_class->stop = gst_edgesrc_stop;
  gstbasesrc_class->get_caps = gst_edgesrc_get_caps;
  gstbasesrc_class->get_times = gst_edgesrc_get_times;
  gstbasesrc_class->is_seekable = gst_edgesrc_is_seekable;
  gstbasesrc_class->create = gst_edgesrc_create;
  gstbasesrc_class->query = gst_edgesrc_query;

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
      GST_EDGE_ELEM_NAME_SRC, 0, "Edge src");
}

/**
 * @brief initialize edgesrc element 
 */
static void
gst_edgesrc_init (GstEdgeSrc * self)
{
  GstBaseSrc *basesrc = GST_BASE_SRC (self);

  gst_base_src_set_format (basesrc, GST_FORMAT_TIME);
  gst_base_src_set_async (basesrc, FALSE);

  /** @todo set default value of props */
}

/**
 * @brief set property
 */
static void
gst_edgesrc_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstEdgeSrc *self = GST_EDGESRC (object);

  switch (prod_id) {
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
gst_edgesrc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstEdgeSrc *self = GST_EDGESRC (object);

  switch (prop_id) {
      /** @todo props */
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the object
 */
static void
gst_edgesrc_class_finalize (GObject * object)
{
  GstEdgeSrc *self = GST_EDGESRC (object);
  /** @todo finalize - free all pointer in element */
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief handle edgesrc's state change
 */
static GstStateChangeReturn
gst_edgesrc_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;

  GstEdgeSrc *self = GST_EDGESRC (element);

  /** @todo handle transition  */

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  /** @todo handle transition */

  return ret;
}

/**
 * @brief start edgesrc, called when state changed null to ready
 */
static gboolean
gst_edgesrc_start (GstBaseSrc * basesrc)
{
  GstEdgeSrc *self = GST_EDGESRC (basesrc);

  /** @todo start */
}

/**
 * @brief stop edgesrc, called when state changed ready to null
 */
static gboolean
gst_edgesrc_stop (GstBaseSrc * basesrc)
{
  GstEdgeSrc *self = GST_EDGESRC (basesrc);

  /** @todo stop */
}

/**
 * @brief Get caps of subclass
 */
static GstCaps *
gst_edgesrc_get_caps (GstBaseSrc * basesrc, GstCaps * filter)
{
  /** @todo get caps */
}

/**
 * @brief Return the time information of the given buffer
 */
static void
gst_edgesrc_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
  /** @todo get times */
}

/**
 * @brief Check if source supports seeking
 */
static gboolean
gst_edgesrc_is_seekable (GstBaseSrc * basesrc)
{
  /** @todo is seekable */
}

/**
 * @brief Create a buffer containing the subscribed data
 */
static GstFlowReturn
gst_edgesrc_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf)
{
  GstEdgeSrc *self = GST_EDGESRC (basesrc);

  /** @todo create */
}

/**
 * @brief An implementation of the GstBaseSrc vmethod that handles queries
 */
static gboolean
gst_edgesrc_query (GstBaseSrc * basesrc, GstQuery * query)
{
  /** @todo query */
}
