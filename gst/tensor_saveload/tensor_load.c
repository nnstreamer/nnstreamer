/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 */
/**
 * @file	tensor_load.c
 * @date	24 Jul 2018
 * @brief	GStreamer plugin to convert other/tensorsave to other/tensor(s)
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
/**
 * SECTION:element-tensor_load
 *
 * Decode other/tensorsave and provide the raw data represented in
 * other/tensor (if # tensors = 1) or other/tensors (if # tensors > 1).
 *
 * @todo NYI: this code supports # tensors = 1 case only!
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m filesrc location="test.tnsr" ! tensor_load ! tensor_filter ... ! tensor_sink ...
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>
#include <tensor_common.h>

#include "tensor_load.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_load_debug);
#define GST_CAT_DEFAULT gst_tensor_load_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT,
};

/**
 * the capabilities of the inputs
 *
 * In v0.0.1, this is "bitmap" image stream
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("other/tensorsave")
    );

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);
    /* This can be both other/tensor and other/tensors */

#define gst_tensor_load_parent_class parent_class
G_DEFINE_TYPE (GstTensor_Load, gst_tensor_load, GST_TYPE_BASE_TRANSFORM);

static void gst_tensor_load_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_load_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_tensor_load_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_tensor_load_transform_ip (GstBaseTransform *
    trans, GstBuffer * buf);
static GstCaps *gst_tensor_load_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_load_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_load_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_load_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
/* GObject vmethod implementations */

/**
 * @brief initialize the tensor_load's class
 */
static void
gst_tensor_load_class_init (GstTensor_LoadClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  GstTensor_LoadClass *klass;

  klass = (GstTensor_LoadClass *) g_class;
  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_load_set_property;
  gobject_class->get_property = gst_tensor_load_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "Tensor_Load",
      "Convert other/tensorsave to other/tensors or other/tensor",
      "Convert other/tensorsave to other/tensors or other/tensor if # tensors is 1",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_load_transform);
  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_tensor_load_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_load_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_load_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_load_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_load_transform_size);
}

/**
 * @brief initialize the new element (G_DEFINE_TYPE requires this)
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_load_init (GstTensor_Load * filter)
{
  filter->silent = TRUE;

  filter->num_tensors = 0;
  filter->dims = NULL;
  filter->types = NULL;
  filter->ranks = NULL;
  filter->framerate_numerator = -1;
  filter->framerate_denominator = -1;
  filter->frameSize = 0;
}

/**
 * @brief Set property (gst element vmethod)
 */
static void
gst_tensor_load_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensor_Load *filter = GST_TENSOR_LOAD (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_load_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensor_Load *filter = GST_TENSOR_LOAD (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/******************************************************************
 * GstElement vmethod implementations
 */

#define return_false_if_fail(x)	\
  ret = (x); \
  if (!ret) \
    return FALSE; \
  ;

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensor_load_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  /** @todo NYI */
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief in-place transform. required vmethod for BaseTransform class.
 */
static GstFlowReturn
gst_tensor_load_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  /** @todo NYI */
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief configure srcpad cap from "proposed" cap. (required vmethod for BaseTransform)
 *
 * @param trans ("this" pointer)
 * @param direction (why do we need this?)
 * @param caps sinkpad cap
 * @param filter this element's cap (don't know specifically.)
 */
static GstCaps *
gst_tensor_load_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  /** @todo NYI */
  return NULL;
}

/**
 * @brief fixate caps. required vmethod of BaseTransform
 */
static GstCaps *
gst_tensor_load_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  /** @todo NYI */
  return NULL;
}

/**
 * @brief set caps. required vmethod of BaseTransform
 */
static gboolean
gst_tensor_load_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  /** @todo NYI */
  return FALSE;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * We cannot directly get the value from size value, we need to review the pad-caps.
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_load_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  /** @todo NYI */
  return FALSE;
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
NNSTREAMER_PLUGIN_INIT (tensor_load)
{
  /**
   * debug category for fltering log messages
   *
   * exchange the string 'Template tensor_load' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_load_debug, "tensor_load",
      0, "Template tensor_load");

  return gst_element_register (plugin, "tensor_load",
      GST_RANK_NONE, GST_TYPE_TENSOR_LOAD);
}

#ifndef SINGLE_BINARY
/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

/**
 * gstreamer looks for this structure to register tensor_loads
 *
 * exchange the string 'Template tensor_load' with your tensor_load description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_load,
    "Load other/tensor_save and as other/tensor or other/tensors",
    gst_tensor_load_plugin_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer/");
#endif
