/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 */

/**
 * @file	gsttesttensors.c
 * @date	26 June 2018
 * @brief	Element to test tensors (witch generates tensors)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-testtensors
 *
 * This is the element to test tensors only.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! testtensors ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "gsttesttensors.h"
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_testtensors_debug);
#define GST_CAT_DEFAULT gst_testtensors_debug

/** Properties */
enum
{
  PROP_0,
  PROP_PASSTHROUGH,
  PROP_SILENT
};

/**
 * the capabilities of the inputs and outputs.
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

#define gst_testtensors_parent_class parent_class
G_DEFINE_TYPE (Gsttesttensors, gst_testtensors, GST_TYPE_ELEMENT);

/** GObject vmethod implementations */
static void gst_testtensors_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_testtensors_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_testtensors_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstFlowReturn gst_testtensors_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);

/**
 * @brief initialize the testtensors's class
 */
static void
gst_testtensors_class_init (GsttesttensorsClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_testtensors_set_property;
  gobject_class->get_property = gst_testtensors_get_property;

  g_object_class_install_property (gobject_class, PROP_PASSTHROUGH,
      g_param_spec_boolean ("passthrough", "Passthrough",
          "Flag to pass incoming buufer", FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          TRUE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "testtensors",
      "Test/Tensor",
      "Get x-raw and push tensors including three tensors",
      "Jijoong Moon <jijoong.moon@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_testtensors_init (Gsttesttensors * filter)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_event_function (filter->sinkpad,
      GST_DEBUG_FUNCPTR (gst_testtensors_sink_event));
  gst_pad_set_chain_function (filter->sinkpad,
      GST_DEBUG_FUNCPTR (gst_testtensors_chain));
  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);

  filter->silent = TRUE;
  filter->passthrough = FALSE;
}

/**
 * @brief Set property vmethod
 */
static void
gst_testtensors_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gsttesttensors *filter = GST_TESTTENSORS (object);

  switch (prop_id) {
    case PROP_PASSTHROUGH:
      filter->passthrough = g_value_get_boolean (value);
      break;
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get property vmethod
 */
static void
gst_testtensors_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gsttesttensors *filter = GST_TESTTENSORS (object);

  switch (prop_id) {
    case PROP_PASSTHROUGH:
      g_value_set_boolean (value, filter->passthrough);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Set Caps in pad.
 * @param filter Gsttesttensors instance
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully generate & set cap, return TRUE)
 */
static gboolean
gst_test_tensors_setcaps (Gsttesttensors * filter, const GstCaps * caps)
{
  GstCaps *othercaps;
  gboolean ret;
  guint i, num_tensors;
  GstTensorConfig config;
  GstTensorsConfig tensors_config;
  GstStructure *s = gst_caps_get_structure (caps, 0);

  g_assert (gst_tensor_config_from_structure (&config, s));
  g_assert (gst_tensor_config_validate (&config));

  filter->in_config = config;

  /* parse config to test tensors */
  gst_tensors_config_init (&tensors_config);

  num_tensors = tensors_config.info.num_tensors = config.info.dimension[0];

  config.info.dimension[0] = 1;
  for (i = 0; i < num_tensors; i++) {
    tensors_config.info.info[i] = config.info;
  }

  tensors_config.rate_n = config.rate_n;
  tensors_config.rate_d = config.rate_d;
  g_assert (gst_tensors_config_validate (&tensors_config));

  filter->out_config = tensors_config;

  othercaps = gst_tensors_caps_from_config (&tensors_config);
  ret = gst_pad_set_caps (filter->srcpad, othercaps);
  gst_caps_unref (othercaps);

  return ret;
}

/**
 * @brief make GstBuffer for output tensor.
 * @param filter Gsttesttensors
 * @param inbuf incomming GstBuffer. (x-raw)
 * @return GstBuffer as 'tensors' with three tensors for test
 */
static GstBuffer *
gst_test_tensors (Gsttesttensors * filter, GstBuffer * inbuf)
{
  GstBuffer *outbuf;
  guint i, num_tensors;
  guint d1, d2;
  guint width, height;
  GstMapInfo src_info;
  size_t span, span1;

  outbuf = gst_buffer_new ();
  gst_buffer_map (inbuf, &src_info, GST_MAP_READ);

  num_tensors = filter->out_config.info.num_tensors;

  for (i = 0; i < num_tensors; i++) {
    GstMapInfo info;
    GstMemory *mem;

    width = filter->out_config.info.info[i].dimension[1];
    height = filter->out_config.info.info[i].dimension[2];

    mem = gst_allocator_alloc (NULL, width * height, NULL);
    gst_memory_map (mem, &info, GST_MAP_WRITE);

    for (d1 = 0; d1 < height; d1++) {
      span = d1 * width * num_tensors;
      span1 = d1 * width;
      for (d2 = 0; d2 < width; d2++) {
        info.data[span1 + d2] = src_info.data[span + (d2 * num_tensors) + i];
      }
    }

    gst_memory_unmap (mem, &info);
    gst_buffer_append_memory (outbuf, mem);
  }

  gst_buffer_unmap (inbuf, &src_info);

  return outbuf;
}

/**
 * @brief this function handles sink events.
 * GstElement vmethod implementations.
 */
static gboolean
gst_testtensors_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  Gsttesttensors *filter;
  gboolean ret;

  filter = GST_TESTTENSORS (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;

      gst_event_parse_caps (event, &caps);
      ret = gst_test_tensors_setcaps (filter, caps);

      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }
  return ret;
}

/**
 * @brief chain function, this function does the actual processing.
 * GstElement vmethod implementations.
 */
static GstFlowReturn
gst_testtensors_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Gsttesttensors *filter;
  GstBuffer *out;

  filter = GST_TESTTENSORS (parent);

  /** just push out the incoming buffer without touching it */
  if (filter->passthrough)
    return gst_pad_push (filter->srcpad, buf);

  out = gst_test_tensors (filter, buf);

  gst_buffer_unref (buf);

  /** In order to keep the data in outbuffer, we have to increase buffer ref count */
  gst_buffer_ref (out);

  return gst_pad_push (filter->srcpad, out);
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
testtensors_init (GstPlugin * plugin)
{
  /**
   * debug category for fltering log messages
   * exchange the string 'Template testtensors' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_testtensors_debug, "testtensors",
      0, "Element testtensors to test tensors");

  return gst_element_register (plugin, "testtensors", GST_RANK_NONE,
      GST_TYPE_TESTTENSORS);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "testtensors"
#endif

/**
 * gstreamer looks for this structure to register testtensorss
 * exchange the string 'Template testtensors' with your testtensors description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    testtensors,
    "Element testtensors to test tensors",
    testtensors_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/")
