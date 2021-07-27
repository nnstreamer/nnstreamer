/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversink.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query server sink
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_query_serversink.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversink_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversink_debug

/**
 * @brief the capabilities of the inputs and outputs.
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_query_serversink_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSink, gst_tensor_query_serversink,
    GST_TYPE_BASE_SINK);

static void gst_tensor_query_serversink_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversink_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversink_finalize (GObject * object);

/**
 * @brief initialize the class
 */
static void
gst_tensor_query_serversink_class_init (GstTensorQueryServerSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gstbasesink_class = (GstBaseSinkClass *) klass;
  gstelement_class = (GstElementClass *) gstbasesink_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_query_serversink_set_property;
  gobject_class->get_property = gst_tensor_query_serversink_get_property;
  gobject_class->finalize = gst_tensor_query_serversink_finalize;

  /** install property goes here */

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSink", "Sink/Tensor/Query",
      "Send tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  /** method override goes here */

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_serversink_debug,
      "tensor_query_serversink", 0, "Tensor Query Server Sink");
}

/** @todo Remove when the dummy functions are implemented. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * @brief initialize the new element
 */
static void
gst_tensor_query_serversink_init (GstTensorQueryServerSink * self)
{
  return;
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_serversink_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set property
 */
static void
gst_tensor_query_serversink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  return;
}

/**
 * @brief get property
 */
static void
gst_tensor_query_serversink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  return;
}

#pragma GCC diagnostic pop
