/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversrc.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query_server src
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_query_serversrc.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversrc_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversrc_debug

/**
 * @brief the capabilities of the inputs and outputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_query_serversrc_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSrc, gst_tensor_query_serversrc,
    GST_TYPE_PUSH_SRC);

static void gst_tensor_query_serversrc_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_finalize (GObject * object);

/**
 * @brief initialize the class
 */
static void
gst_tensor_query_serversrc_class_init (GstTensorQueryServerSrcClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *gstbasesrc_class;
  GstPushSrcClass *gstpushsrc_class;

  gstpushsrc_class = (GstPushSrcClass *) klass;
  gstbasesrc_class = (GstBaseSrcClass *) gstpushsrc_class;
  gstelement_class = (GstElementClass *) gstbasesrc_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_query_serversrc_set_property;
  gobject_class->get_property = gst_tensor_query_serversrc_get_property;
  gobject_class->finalize = gst_tensor_query_serversrc_finalize;

  /** install property goes here */

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSrc", "Source/Tensor/Query",
      "Receive tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  /** method override goes here */

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_serversrc_debug,
      "tensor_query_serversrc", 0, "Tensor Query Server Source");
}

/**
 * @brief initialize the new element
 */
static void
gst_tensor_query_serversrc_init (GstTensorQueryServerSrc * self)
{
  return;
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_serversrc_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief set property
 */
static void
gst_tensor_query_serversrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  return;
}

/**
 * @brief get property
 */
static void
gst_tensor_query_serversrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  return;
}
