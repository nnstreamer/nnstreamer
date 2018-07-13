/**
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
 * @file	gsttensorscheck.c
 * @date	26 June 2018
 * @brief	test element to check tensors
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @bug         no known bugs
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 *
 */

/**
 * SECTION:element-tensorscheck
 *
 * This is the element to test tensors only.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensorscheck ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>

#include "gsttensorscheck.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensorscheck_debug);
#define GST_CAT_DEFAULT gst_tensorscheck_debug

/** Filter signals and args */
enum
{
  /** FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

/** the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

#define gst_tensorscheck_parent_class parent_class
G_DEFINE_TYPE (Gsttensorscheck, gst_tensorscheck, GST_TYPE_ELEMENT);

static void gst_tensorscheck_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensorscheck_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_tensorscheck_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstFlowReturn gst_tensorscheck_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);

/** GObject vmethod implementations */

/**
 * @brief initialize the tensorscheck's class
 */
static void
gst_tensorscheck_class_init (GsttensorscheckClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_tensorscheck_set_property;
  gobject_class->get_property = gst_tensorscheck_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "tensorscheck",
      "Test Tensors",
      "Get Tensors and Re-construct tensor to check",
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
gst_tensorscheck_init (Gsttensorscheck * filter)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_event_function (filter->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensorscheck_sink_event));
  gst_pad_set_chain_function (filter->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensorscheck_chain));
  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);

  filter->silent = FALSE;
}

/**
 * @brief set property vmthod
 */
static void
gst_tensorscheck_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gsttensorscheck *filter = GST_TENSORSCHECK (object);

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
 * @brief get property vmthod
 */
static void
gst_tensorscheck_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gsttensorscheck *filter = GST_TENSORSCHECK (object);

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
tensorscheck_init (GstPlugin * tensorscheck)
{
  /** debug category for fltering log messages
   *
   * exchange the string 'Template tensorscheck' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensorscheck_debug, "tensorscheck",
      0, "Template tensorscheck");

  return gst_element_register (tensorscheck, "tensorscheck", GST_RANK_NONE,
      GST_TYPE_TENSORSCHECK);
}

/** PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensorscheck"
#endif

/** gstreamer looks for this structure to register tensorschecks
 *
 * exchange the string 'Template tensorscheck' with your tensorscheck description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensorscheck,
    "Template tensorscheck",
    tensorscheck_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/")

/**
 * @brief Set Caps in pad.
 * @param filter Gsttensorscheck instance
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully generate & set cap, return TRUE)
 */
     static gboolean
         gst_tensors_check_setcaps (Gsttensorscheck * filter, GstCaps * caps)
{
  gboolean ret;
  gint dim;
  GstCaps *othercaps;
  const gchar *dim_string;
  const gchar *types;
  int i;

  GstStructure *s = gst_caps_get_structure (caps, 0);
  gst_structure_get_int (s, "rank", &filter->rank);
  gst_structure_get_int (s, "num_tensors", &dim);
  filter->num_tensors = dim;

  dim_string = gst_structure_get_string (s, "dimensions");
  if (dim_string) {
    debug_print (!filter->silent, "dimension sting : %s\n", dim_string);
    filter->dimensions = parse_dimensions (dim_string);
    for (i = 0; i < filter->num_tensors; i++) {
      tensor_dim *d = g_array_index (filter->dimensions, tensor_dim *, i);
      debug_print (!filter->silent, "dimensions[%d] %d %d %d %d\n", i, (*d)[0],
          (*d)[1], (*d)[2], (*d)[3]);
    }
  } else {
    err_print ("Cannot get dimensions for negotiation!\n");
  }
  gst_structure_get_fraction (s, "framerate", &filter->framerate_numerator,
      &filter->framerate_denominator);
  types = gst_structure_get_string (s, "types");
  if (types) {
    debug_print (!filter->silent, "types string : %s\n", types);
    filter->types = parse_types (types);
    for (i = 0; i < filter->num_tensors; i++) {
      tensor_type *t = g_array_index (filter->types, tensor_type *, i);
      debug_print (!filter->silent, "types[%d] %s\n", i,
          tensor_element_typename[(*t)]);
    }
  } else {
    err_print ("Cannot get types for negotiation!\n");
  }

  tensor_dim *d = g_array_index (filter->dimensions, tensor_dim *, 0);
  tensor_type *t = g_array_index (filter->types, tensor_type *, 0);
  othercaps = gst_caps_new_simple ("other/tensor",
      "rank", G_TYPE_INT, filter->rank,
      "dim1", G_TYPE_INT, (*d)[0],
      "dim2", G_TYPE_INT, (*d)[1],
      "dim3", G_TYPE_INT, (*d)[2],
      "dim4", G_TYPE_INT, (*d)[3],
      "type", G_TYPE_STRING, tensor_element_typename[*t],
      "framerate", GST_TYPE_FRACTION, filter->framerate_numerator,
      filter->framerate_denominator, NULL);
  ret = gst_pad_set_caps (filter->srcpad, othercaps);
  gst_caps_unref (othercaps);
  return ret;
}

/**
 * @brief make GstBuffer for output tensor.
 * @param filter Gsttensorscheck
 * @param inbuf incomming GstBuffer. (tensors)
 * @return GstBuffer one big GstBuffer
 */
static GstBuffer *
gst_tensors_check (Gsttensorscheck * filter, GstBuffer * inbuf)
{
  GstBuffer *outbuf;
  gint num_tensors;
  GstMapInfo info, src_info, dest_info;
  GstMemory *buffer_mem;
  tensor_dim *dim;
  unsigned int d0, d1, d2, i;
  gboolean ret;

  /** Mapping input buffer (tensors) into src_info */
  gst_buffer_map (inbuf, &src_info, GST_MAP_READ);

  /** Making output GstBuffer */
  outbuf = gst_buffer_new ();

  /** Making output buffer (one big buffer for check tensors) */
  buffer_mem = gst_allocator_alloc (NULL,
      /** filter->dimension[0] * filter->dimension[1] * filter->dimension[2] * */
      /** filter->dimension[3], NULL); */
      3 * 640 * 480 * 1, NULL);
  gst_buffer_append_memory (outbuf, buffer_mem);

  gst_buffer_map (outbuf, &dest_info, GST_MAP_WRITE);

  /** Get number of tensors */
  num_tensors = gst_get_num_tensors (inbuf);
  debug_print (!filter->silent, "Number of Tensors : %d\n", num_tensors);
  for (i = 0; i < num_tensors; i++) {
    GstMemory *mem = gst_get_tensor (inbuf, i);
    if (!mem)
      debug_print (!filter->silent, "Cannot get memory\n");
    dim = gst_get_tensordim (inbuf, i);
    ret = gst_memory_map (mem, &info, GST_MAP_WRITE);
    if (!ret) {
      debug_print (!filter->silent, "Cannot map memory\n");
      return NULL;
    }
    size_t span = 0;
    size_t span1 = 0;
    for (d0 = 0; d0 < (*dim)[3]; d0++) {
      for (d1 = 0; d1 < (*dim)[2]; d1++) {
        span = d1 * (*dim)[1];
        span1 = d1 * (*dim)[1] * 3;
        for (d2 = 0; d2 < (*dim)[1]; d2++) {
          dest_info.data[span1 + (d2 * num_tensors) + i] = info.data[span + d2];
        }
      }
    }
    gst_memory_unmap (mem, &info);
  }
  gst_buffer_unmap (inbuf, &src_info);
  gst_buffer_unmap (outbuf, &dest_info);

  return outbuf;
}

/** GstElement vmethod implementations */

/**
 * @brief this function handles sink events
 */
static gboolean
gst_tensorscheck_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  Gsttensorscheck *filter;
  gboolean ret;

  filter = GST_TENSORSCHECK (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;

      gst_event_parse_caps (event, &caps);
      ret = gst_tensors_check_setcaps (filter, caps);
      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }
  return ret;
}

/**
 * @brief chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_tensorscheck_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  Gsttensorscheck *filter;
  GstBuffer *out;

  filter = GST_TENSORSCHECK (parent);

  if (filter->passthrough)
    return gst_pad_push (filter->srcpad, buf);

  out = gst_tensors_check (filter, buf);
  gst_buffer_unref (buf);

  return gst_pad_push (filter->srcpad, out);
}
