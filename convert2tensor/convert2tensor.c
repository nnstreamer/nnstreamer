/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	convert2tensor.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 */

/**
 * SECTION:element-convert2tensor
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! convert2tensor ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>

#include "convert2tensor.h"

GST_DEBUG_CATEGORY_STATIC (gst_convert2tensor_debug);
#define GST_CAT_DEFAULT gst_convert2tensor_debug

/* Filter signals and args */
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

/* the capabilities of the inputs
 *
 * In v0.0.1, this is "bitmap" image stream
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw, format = (string)RGB")
    );

/* the capabilities of the outputs
 *
 * In v0.0.1, this is 3-d tensor, [color][height][width]
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("other/tensor, "
                       "rank = (int) [ 1, 4 ], "
		       "dimension = (int) [ 0, 65536 ], (int) [ 0, 65536 ], (int) [ 0, 65536 ], (int) [0, 65536 ], "
		       "type = (string) { float32, float64, int32, uint32, int16, uint16, int8, uint8 }, "
		       "framerate = (fraction) [ 0, 1024 ], ")
    );

#define gst_convert2tensor_parent_class parent_class
G_DEFINE_TYPE (GstConvert2Tensor, gst_convert2tensor, GST_TYPE_ELEMENT);

static void gst_convert2tensor_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_convert2tensor_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_convert2tensor_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);
static GstFlowReturn gst_convert2tensor_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);

/* GObject vmethod implementations */

/* initialize the convert2tensor's class */
static void
gst_convert2tensor_class_init (GstConvert2TensorClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_convert2tensor_set_property;
  gobject_class->get_property = gst_convert2tensor_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple(gstelement_class,
    "Convert2Tensor",
    "Convert media stream to tensor stream",
    "Converts audio or video stream to tensor stream for neural network framework filters",
    "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_convert2tensor_init (GstConvert2Tensor * filter)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_event_function (filter->sinkpad,
                              GST_DEBUG_FUNCPTR(gst_convert2tensor_sink_event));
  gst_pad_set_chain_function (filter->sinkpad,
                              GST_DEBUG_FUNCPTR(gst_convert2tensor_chain));
  GST_PAD_SET_PROXY_CAPS (filter->sinkpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  GST_PAD_SET_PROXY_CAPS (filter->srcpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);

  filter->silent = FALSE;
}

static void
gst_convert2tensor_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstConvert2Tensor *filter = GST_CONVERT2TENSOR (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_convert2tensor_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstConvert2Tensor *filter = GST_CONVERT2TENSOR (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* GstElement vmethod implementations */

/* this function handles sink events */
static gboolean
gst_convert2tensor_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstConvert2Tensor *filter;
  gboolean ret;

  filter = GST_CONVERT2TENSOR (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps * caps;

      gst_event_parse_caps (event, &caps);
      /* do something with the caps */

      /* and forward */
      ret = gst_pad_event_default (pad, parent, event);
      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }
  return ret;
}

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_convert2tensor_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstConvert2Tensor *filter;

  filter = GST_CONVERT2TENSOR (parent);

  if (filter->silent == FALSE)
    g_print ("I'm plugged, therefore I'm in.\n");

  /* just push out the incoming buffer without touching it */
  return gst_pad_push (filter->srcpad, buf);
}


/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
convert2tensor_init (GstPlugin * convert2tensor)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template convert2tensor' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_convert2tensor_debug, "convert2tensor",
      0, "Template convert2tensor");

  return gst_element_register (convert2tensor, "convert2tensor", GST_RANK_NONE,
      GST_TYPE_CONVERT2TENSOR);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirstconvert2tensor"
#endif

/* gstreamer looks for this structure to register convert2tensors
 *
 * exchange the string 'Template convert2tensor' with your convert2tensor description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    convert2tensor,
    "Template convert2tensor",
    convert2tensor_init,
    VERSION,
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)
