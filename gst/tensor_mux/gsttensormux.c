/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
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
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	gsttensormux.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to mux tensors (as a filter for other general neural network filters)
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>

#include "gsttensormux.h"
#include <tensor_meta.h>

GST_DEBUG_CATEGORY_STATIC (gst_tensormux_debug);
#define GST_CAT_DEFAULT gst_tensormux_debug

enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

/**
 * @brief the capabilities of the inputs and outputs.
 * describe the real formats here.
 */
static GstStaticPadTemplate src_templ = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

static GstStaticPadTemplate sink_templ = GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

#define gst_tensor_mux_parent_class parent_class
G_DEFINE_TYPE (GstTensorMux, gst_tensor_mux, GST_TYPE_ELEMENT);

static gboolean
gst_tensor_mux_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);

static void gst_tensor_mux_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_mux_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);

static GstPad *gst_tensor_mux_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps);
static void gst_tensor_mux_release_pad (GstElement * element, GstPad * pad);

static GstFlowReturn
gst_tensor_mux_chain (GstPad * pad, GstObject * object, GstBuffer * buffer);

static gboolean
gst_tensor_mux_pad_setcaps (GstPad * pad, GstObject * parent, GstCaps * caps);

/**
 * @brief initialize the tensor_mux's class
 */
static void
gst_tensor_mux_class_init (GstTensorMuxClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->get_property = gst_tensor_mux_get_property;
  gobject_class->set_property = gst_tensor_mux_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gstelement_class->request_new_pad = gst_tensor_mux_request_new_pad;
  gstelement_class->release_pad = gst_tensor_mux_release_pad;

  gst_element_class_set_details_simple (gstelement_class,
      "tensormux",
      "Mux multiple tensor stream",
      "Merge multiple tensor stream to tensors stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_mux_init (GstTensorMux * mux)
{
  mux->srcpad = gst_pad_new_from_static_template (&src_templ, "src");
  gst_element_add_pad (GST_ELEMENT (mux), mux->srcpad);
  mux->num_tensors = 0;
  mux->byte_count = 0;
  mux->silent = FALSE;
}


/**
 * @brief making new request pad (gst element vmethod)
 */
static GstPad *
gst_tensor_mux_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * name, const GstCaps * caps)
{
  GstPad *pad;
  pad = gst_pad_new_from_template (templ, name);

  gst_pad_set_event_function (pad,
      GST_DEBUG_FUNCPTR (gst_tensor_mux_sink_event));
  gst_pad_set_chain_function (pad, GST_DEBUG_FUNCPTR (gst_tensor_mux_chain));

  gst_element_add_pad (element, pad);
  return pad;
}

/**
 * @brief release request pad (gst element vmethod)
 */
static void
gst_tensor_mux_release_pad (GstElement * element, GstPad * pad)
{
  gst_element_remove_pad (element, pad);
}

/**
 * @brief event function (gst element vmethod)
 */
static gboolean
gst_tensor_mux_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  gboolean ret = TRUE;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      ret = gst_tensor_mux_pad_setcaps (pad, parent, caps);
      break;
    }
    default:
      break;
  }

  if (!ret)
    return FALSE;

  return ret;
}

/**
 * @brief set caps (gst element vmethod)
 */
static gboolean
gst_tensor_mux_pad_setcaps (GstPad * pad, GstObject * parent, GstCaps * caps)
{
  GstTensorMux *mux = GST_TENSOR_MUX (parent);
  gboolean ret = TRUE;
  GstStructure *s;

  s = gst_caps_get_structure (caps, 0);
  if (strcmp (gst_structure_get_name (s), "other/tensor") != 0)
    ret = FALSE;

  gst_object_unref (mux);
  return ret;
}


/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_mux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorMux *filter = GST_TENSOR_MUX (object);

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
gst_tensor_mux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorMux *filter = GST_TENSOR_MUX (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
tensor_mux_init (GstPlugin * tensormux)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template tensor_mux' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensormux_debug, "tensormux", 0, "Tensor Muxer");

  return gst_element_register (tensormux, "tensormux",
      GST_RANK_NONE, GST_TYPE_TENSOR_MUX);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensormux"
#endif

/* gstreamer looks for this structure to register tensor_converters
 *
 * exchange the string 'Template tensor_converter' with your tensor_converter description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensormux,
    "tensormux",
    tensor_mux_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");

/**
 * @brief chain function (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_mux_chain (GstPad * pad, GstObject * object, GstBuffer * buffer)
{
  gst_buffer_unref (buffer);
  return GST_FLOW_OK;
}
