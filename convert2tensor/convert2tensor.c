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
 *
 * @TODO Note: I'm not sure of this.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("other/tensor, "
                       "rank = (uint) [ 1, 4 ], "
		       "dim1 = (uint) [ 1, 65535 ], "
		       "dim2 = (uint) [ 1, 65535 ], "
		       "dim3 = (uint) [ 1, 65535 ], "
		       "dim4 = (uint) [ 1, 65535 ], "
		       "type = (string) { float32, float64, int32, uint32, int16, uint16, int8, uint8 }, "
		       "framerate = (fraction) [ 0, 1024 ], ")
    );

#define gst_convert2tensor_parent_class parent_class
G_DEFINE_TYPE (GstConvert2Tensor, gst_convert2tensor, GST_TYPE_BASE_TRANSFORM);

static void gst_convert2tensor_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_convert2tensor_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_convert2tensor_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf);
static GstFlowReturn gst_convert2tensor_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf);
static GstCaps* gst_convert2tensor_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps,
						  GstCaps *filter);
static GstCaps* gst_convert2tensor_fixate_caps(GstBaseTransform *trans,
                                               GstPadDirection direction,
					       GstCaps *caps,
					       GstCaps *othercaps);
static gboolean gst_convert2tensor_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps,
					    GstCaps *outcaps);
static gboolean gst_convert2tensor_transform_size(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps, gsize size,
						  GstCaps *othercpas, gsize *othersize);
static gboolean gst_convert2tensor_get_unit_size(GstBaseTransform *trans,
                                                 GstCaps *caps, gsize *size);
/* GObject vmethod implementations */

/* initialize the convert2tensor's class */
static void
gst_convert2tensor_class_init (GstConvert2TensorClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  GstConvert2TensorClass *klass;

  klass = (GstConvert2TensorClass *) g_class;
  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

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

  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR(gst_convert2tensor_transform);
  trans_class->transform_ip = GST_DEBUG_FUNCPTR(gst_convert2tensor_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps = GST_DEBUG_FUNCPTR(gst_convert2tensor_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR(gst_convert2tensor_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR(gst_convert2tensor_set_caps);

  /* Allocation units */
  trans_class->transform_size = GST_DEBUG_FUNCPTR(gst_convert2tensor_transform_size);
  trans_class->get_unit_size = GST_DEBUG_FUNCPTR(gst_convert2tensor_get_unit_size);
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_convert2tensor_init (GstConvert2Tensor * filter)
{
  filter->silent = FALSE;
  filter->tensorConfigured = FALSE;
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

/******************************************************************
 * GstElement vmethod implementations
 */

#define return_false_if_fail(x)	\
  ret = (x); \
  if (!ret) \
    return FALSE; \
  ;
/* Configure tensor metadata from sink caps */
static gboolean
gst_convert2tensor_configure_tensor(const GstCaps *caps, GstConvert2Tensor *filter) {
  GstStructure *structure;
  gint rank;
  gint dimension[GST_CONVERT2TENSOR_TENSOR_RANK_LIMIT];
  tensor_type type;
  gint framerate_numerator;
  gint framerate_denominator;
  gboolean ret;
  GstCaps *outcaps;
  int i;

  /* This caps is coming from video/x-raw */
  structure = gst_caps_get_structure(caps, 0);
  rank = 3; /* [color-space][height][width] */
  return_false_if_fail(gst_structure_get_int(structure, "width", &dimension[0]));
  return_false_if_fail(gst_structure_get_int(structure, "height", &dimension[1]));
  return_false_if_fail(gst_structure_get_fraction(structure, "framerate", &framerate_numerator, &framerate_denominator));
  type = _C2T_UINT8; /* Assume color depth per component is 8 bit */
  dimension[2] = 3; /* R G B */
  dimension[3] = 1; /* This is 3-D Tensor */
  /* Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

  if (filter->tensorConfigured == TRUE) {
    /* It has been already configured. Check if they are consistent */
    if (rank == filter->rank &&
	type == filter->type &&
	framerate_numerator == filter->framerate_numerator &&
	framerate_denominator == filter->framerate_denominator) {
      for (i = 0; i < GST_CONVERT2TENSOR_TENSOR_RANK_LIMIT; i++)
        if (dimension[i] != filter->dimension[i])
	  return FALSE;
      return TRUE;
    }
    g_printerr("  Something's wrong. The tensor metadata is inconsistent.\n");
    return FALSE;
  }

  filter->rank = rank;
  for (i = 0; i < GST_CONVERT2TENSOR_TENSOR_RANK_LIMIT; i++)
    filter->dimension[i] = dimension[i];
  filter->type = type;
  filter->framerate_numerator = framerate_numerator;
  filter->framerate_denominator = framerate_denominator;

  filter->tensorConfigured = TRUE;
  return TRUE;
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


static GstFlowReturn gst_convert2tensor_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf)
{
}

static GstFlowReturn gst_convert2tensor_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf)
{
}

static GstCaps* gst_convert2tensor_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps,
						  GstCaps *filter)
{
}

static GstCaps* gst_convert2tensor_fixate_caps(GstBaseTransform *trans,
                                               GstPadDirection direction,
					       GstCaps *caps,
					       GstCaps *othercaps)
{
}

static gboolean gst_convert2tensor_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps,
					    GstCaps *outcaps)
{
}

static gboolean gst_convert2tensor_transform_size(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps, gsize size,
						  GstCaps *othercpas, gsize *othersize)
{
}

static gboolean gst_convert2tensor_get_unit_size(GstBaseTransform *trans,
                                                 GstCaps *caps, gsize *size)
{
}
