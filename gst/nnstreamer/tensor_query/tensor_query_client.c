/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_client.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query client
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_query_client.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_client_debug);
#define GST_CAT_DEFAULT gst_tensor_query_client_debug

/**
 * @brief Default caps string for pads.
 */
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_DEFAULT ";" GST_TENSORS_FLEX_CAP_DEFAULT

/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief the capabilities of the outputs.
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

#define gst_tensor_query_client_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryClient, gst_tensor_query_client,
    GST_TYPE_BASE_TRANSFORM);

/* GObject vmethod implementations */
static void gst_tensor_query_client_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_client_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_client_finalize (GObject * object);

/**
 * @brief initialize the class
 */
static void
gst_tensor_query_client_class_init (GstTensorQueryClientClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_query_client_set_property;
  gobject_class->get_property = gst_tensor_query_client_get_property;
  gobject_class->finalize = gst_tensor_query_client_finalize;

  /** install property goes here */

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryClient", "Filter/Tensor/Query",
      "Handle querying tensor data through the network",
      "Samsung Electronics Co., Ltd.");

  /** method override goes here */

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_client_debug, "tensor_query_client",
      0, "Tensor Query Client");
}

/**
 * @brief initialize the new element
 */
static void
gst_tensor_query_client_init (GstTensorQueryClient * self)
{
  return;
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_client_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set property
 */
static void
gst_tensor_query_client_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  return;
}

/**
 * @brief get property
 */
static void
gst_tensor_query_client_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  return;
}
