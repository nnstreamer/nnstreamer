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
 */
/**
 * @file	gsttesttensors.c
 * @date	26 June 2018
 * @brief	Element to test tensors (witch generates tensors)
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 *
 */
/**
 * SECTION:element-testtensors
 *
 * FIXME:Describe testtensors here.
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

#include <gst/gst.h>

#include "gsttesttensors.h"
#include <string.h>

GST_DEBUG_CATEGORY_STATIC (gst_testtensors_debug);
#define GST_CAT_DEFAULT gst_testtensors_debug

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

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS
    ("video/x-raw, format = (string) {RGB, BGRx}, views = (int)1, interlace-mode = (string)progressive, framerate = (fraction)[ 0/1, 2147483647/1 ]")
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

#define gst_testtensors_parent_class parent_class
G_DEFINE_TYPE (Gsttesttensors, gst_testtensors, GST_TYPE_ELEMENT);

static void gst_testtensors_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_testtensors_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_testtensors_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstFlowReturn gst_testtensors_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);

/* GObject vmethod implementations */

/* initialize the testtensors's class */
static void
gst_testtensors_class_init (GsttesttensorsClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_testtensors_set_property;
  gobject_class->get_property = gst_testtensors_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "testtensors",
      "Test Tensors",
      "Get x-raw and push tensors including three tensors",
      "Jijoong Moon <jijoong.moon@samsung.com>");

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

  filter->silent = FALSE;
}

static void
gst_testtensors_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gsttesttensors *filter = GST_TESTTENSORS (object);

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
gst_testtensors_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gsttesttensors *filter = GST_TESTTENSORS (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
testtensors_init (GstPlugin * testtensors)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template testtensors' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_testtensors_debug, "testtensors",
      0, "Template testtensors");

  return gst_element_register (testtensors, "testtensors", GST_RANK_NONE,
      GST_TYPE_TESTTENSORS);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "testtensors"
#endif

/* gstreamer looks for this structure to register testtensorss
 *
 * exchange the string 'Template testtensors' with your testtensors description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    testtensors,
    "Template testtensors",
    testtensors_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/")

/*
 * @brief Set Caps in pad.
 * @param filter Gsttesttensors instance
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully generate & set cap, return TRUE)
 */
     static gboolean
         gst_test_tensors_setcaps (Gsttesttensors * filter, GstCaps * caps)
{
  const gchar *format;
  gint dim;
  GstCaps *othercaps;
  gboolean ret;
  unsigned int i;

  GstStructure *s = gst_caps_get_structure (caps, 0);

  format = gst_structure_get_string (s, "format");
  if (!g_strcmp0 (format, "RGB"))
    filter->num_tensors = 3;
  else
    filter->num_tensors = 4;
  gst_structure_get_int (s, "width", &dim);
  filter->width = dim;
  gst_structure_get_int (s, "height", &dim);
  filter->height = dim;
  gst_structure_get_fraction (s, "framerate", &filter->framerate_numerator,
      &filter->framerate_denominator);
  filter->type = _NNS_UINT8;
  filter->rank = 3;

  char str[256];
  strcpy (str, tensor_element_typename[filter->type]);
  for (i = 1; i < filter->num_tensors; i++) {
    strcat (str, ",");
    strcat (str, tensor_element_typename[filter->type]);
  }

  othercaps = gst_caps_new_simple ("other/tensors",
      "rank", G_TYPE_INT, filter->rank,
      "num_tensors", G_TYPE_INT, filter->num_tensors,
      "types", G_TYPE_STRING, str,
      "framerate", GST_TYPE_FRACTION, filter->framerate_numerator,
      filter->framerate_denominator, "dimensions", G_TYPE_STRING,
      "1:640:480:1 ,1:640:480:3 ,1:640:480:1", NULL);
  ret = gst_pad_set_caps (filter->srcpad, othercaps);
  gst_caps_unref (othercaps);
  return ret;
}

/*
 * @brief make GstBuffer for output tensor.
 * @param filter Gsttesttensors
 * @param inbuf incomming GstBuffer. (x-raw)
 * @return GstBuffer as 'tensors' with three tensors for test
 */
static GstBuffer *
gst_test_tensors (Gsttesttensors * filter, GstBuffer * inbuf)
{
  GstBuffer *outbuf;
  gint num_tensor;
  GstMapInfo src_info;

  int d1, d2;
  outbuf = gst_buffer_new ();
  gst_buffer_map (inbuf, &src_info, GST_MAP_READ);

  gst_make_tensors (outbuf);

  for (num_tensor = 0; num_tensor < filter->num_tensors; num_tensor++) {
    GstMapInfo info;
    GstMemory *mem = gst_allocator_alloc (NULL, filter->width * filter->height,
        NULL);

    gst_memory_map (mem, &info, GST_MAP_WRITE);
    size_t span = 0;
    size_t span1 = 0;

    for (d1 = 0; d1 < filter->height; d1++) {
      span = d1 * filter->width * filter->num_tensors;
      span1 = d1 * filter->width;
      for (d2 = 0; d2 < filter->width; d2++) {
        info.data[span1 + d2] =
            src_info.data[span + (d2 * filter->num_tensors) + num_tensor];
      }
    }

    tensor_dim dim;
    dim[0] = 1;
    dim[1] = filter->width;
    dim[2] = filter->height;
    dim[3] = 1;

    gst_memory_unmap (mem, &info);
    gst_append_tensor (outbuf, mem, &dim);
  }

  gst_buffer_unmap (inbuf, &src_info);

  return outbuf;
}

/* GstElement vmethod implementations */

/* this function handles sink events */
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

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_testtensors_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Gsttesttensors *filter;
  GstBuffer *out;

  filter = GST_TESTTENSORS (parent);

  /* just push out the incoming buffer without touching it */
  if (filter->passthrough)
    return gst_pad_push (filter->srcpad, buf);

  out = gst_test_tensors (filter, buf);

  gst_buffer_unref (buf);

  /* In order to keep the data in outbuffer, we have to increase buffer ref count */
  gst_buffer_ref (out);

  return gst_pad_push (filter->srcpad, out);
}
