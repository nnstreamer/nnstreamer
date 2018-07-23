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
 *
 */
/**
 * @file	gsttensordemux.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to demux tensors (as a filter for other general neural network filters)
 * @bug         No known bugs
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensordemux
 *
 * A Deuxer that demux tensors stream to tensor stream for NN frameworks.
 * The outputs are always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * ]|
 *
 * </refsect2>
 *
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>

#include "gsttensordemux.h"
#include <tensor_meta.h>

GST_DEBUG_CATEGORY_STATIC (gst_tensor_demux_debug);
#define GST_CAT_DEFAULT gst_tensor_demux_debug

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_SINGLE_STREAM
};

/**
 * @brief the capabilities of the inputs and outputs.
 * describe the real formats here.
 */
static GstStaticPadTemplate src_templ = GST_STATIC_PAD_TEMPLATE ("src_%u",
    GST_PAD_SRC,
    GST_PAD_SOMETIMES,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static GstStaticPadTemplate sink_templ = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

static GstFlowReturn gst_tensor_demux_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);
static gboolean gst_tensor_demux_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstStateChangeReturn gst_tensor_demux_change_state (GstElement * element,
    GstStateChange transition);
static void gst_tensor_demux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_demux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_demux_dispose (GObject * object);
#define gst_tensor_demux_parent_class parent_class
G_DEFINE_TYPE (GstTensorDemux, gst_tensor_demux, GST_TYPE_ELEMENT);


/**
 * @brief initialize the tensor_demux's class
 */
static void
gst_tensor_demux_class_init (GstTensorDemuxClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->dispose = gst_tensor_demux_dispose;
  gobject_class->get_property = gst_tensor_demux_get_property;
  gobject_class->set_property = gst_tensor_demux_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_demux_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "tensordemux",
      "Demux other/tensors stream",
      "Demux tensors stream to other/tensor stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_demux_init (GstTensorDemux * tensor_demux)
{
  tensor_demux->sinkpad =
      gst_pad_new_from_static_template (&sink_templ, "sink");
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_demux), tensor_demux->sinkpad);
  gst_pad_set_chain_function (tensor_demux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_demux_chain));
  gst_pad_set_event_function (tensor_demux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_demux_event));

  tensor_demux->num_tensors = 0;
  tensor_demux->silent = FALSE;
  tensor_demux->singleStream = FALSE;
}

/**
 * @brief function to remove srcpad listfor sink (gst element vmethod)
 */
static void
gst_tensor_demux_remove_src_pads (GstTensorDemux * tensor_demux)
{
  while (tensor_demux != NULL) {
    GstTensorPad *tensor_pad = tensor_demux->srcpads->data;
    gst_element_remove_pad (GST_ELEMENT (tensor_demux), tensor_pad->pad);
    g_free (tensor_pad);
    tensor_demux->srcpads =
        g_slist_delete_link (tensor_demux->srcpads, tensor_demux->srcpads);
  }
  tensor_demux->srcpads = NULL;
  tensor_demux->num_tensors = 0;
}

/**
 * @brief dispose function for sink (gst element vmethod)
 */
static void
gst_tensor_demux_dispose (GObject * object)
{
  GstTensorDemux *tensor_demux = GST_TENSOR_DEMUX (object);

  gst_tensor_demux_remove_src_pads (tensor_demux);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief event function for sink (gst element vmethod)
 */
static gboolean
gst_tensor_demux_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorDemux *tensor_demux;
  tensor_demux = GST_TENSOR_DEMUX (parent);
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      if (!tensor_demux->srcpads) {
        GST_ELEMENT_ERROR (tensor_demux, STREAM, WRONG_TYPE,
            ("This stream contains no valid stremas."),
            ("Got EOS before adding any pads"));
        gst_event_unref (event);
        return FALSE;
      } else {
        return gst_pad_event_default (pad, parent, event);
      }
      break;
    default:
      return gst_pad_event_default (pad, parent, event);
  }
}

/**
 * @brief chain function for sink (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_demux_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstFlowReturn res = GST_FLOW_OK;

  /* GstTensorDemux *tensor_demux; */
  /* tensor_demux = GST_TENSOR_DEMUX (parent); */

  /* NYI */

  return res;
}

/**
 * @brief change state (gst element vmethod)
 */
static GstStateChangeReturn
gst_tensor_demux_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorDemux *tensor_demux;
  GstStateChangeReturn ret;
  tensor_demux = GST_TENSOR_DEMUX (element);

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return ret;

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_tensor_demux_remove_src_pads (tensor_demux);
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_demux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDemux *filter = GST_TENSOR_DEMUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_SINGLE_STREAM:
      filter->singleStream = g_value_get_boolean (value);
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
gst_tensor_demux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDemux *filter = GST_TENSOR_DEMUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_SINGLE_STREAM:
      g_value_set_boolean (value, filter->singleStream);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensordemux"
#endif

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
gboolean
gst_tensor_demux_plugin_init (GstPlugin * tensordemux)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_mux' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_demux_debug, "tensordemux", 0,
      "Tensor Demuxer");
  return gst_element_register (tensordemux, "tensordemux",
      GST_RANK_NONE, GST_TYPE_TENSOR_DEMUX);
}

/**
 * @brief gstreamer looks for this structure to register tensormux
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensordemux,
    "tensordemux",
    gst_tensor_demux_plugin_init, VERSION, "LGPL", "GStreamer",
    "http://gstreamer.net/");
