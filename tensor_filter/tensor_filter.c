/**
 * GStreamer Tensor_Filter
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
 * @file	tensor_filter.c
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This is the main plugin for per-NN-framework plugins.
 * Specific implementations for each NN framework must be written
 * in each framework specific files; e.g., tensor_fitler_tensorflow_lite.c
 *
 */

/**
 * SECTION:element-tensor_filter
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_filter framework=tensorflow-lite, model=./inception_v3.pb, input=3,224,224, output=1000 ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 *
 * If input is other/tensor C array input[1][224][224][3] and
 * output is other/tensor C array output[1][1][1][1000]
 */
/* @TODO I don't sure if we can give "224x224x3" as property values for gstreamer. */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>
#include <glib.h>
#include <glib/gprintf.h>

#include "tensor_filter.h"

GstTensor_Filter_Framework *tensor_filter_supported[] = {
  &NNS_support_tensorflow_lite,
};
const char* nnfw_names[] = {
  "Not supported",

  "custom",
  "tensorflow-lite",
  "tensorflow",
  "caffe2",

  0,
};

GST_DEBUG_CATEGORY_STATIC (gst_tensor_filter_debug);
#define GST_CAT_DEFAULT gst_tensor_filter_debug

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
  PROP_FRAMEWORK,
  PROP_MODEL,
  PROP_INPUT,
  PROP_OUTPUT,
};

/**
 * @brief The capabilities of the inputs
 *
 * @TODO I'm not sure if the range is to be 1, 65535 or larger
 *
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("other/tensor, "
                       "rank = (int) [ 1, 65535 ], "
                       "dim1 = (int) [ 1, 65535 ], "
                       "dim2 = (int) [ 1, 65535 ], "
                       "dim3 = (int) [ 1, 65535 ], "
                       "dim4 = (int) [ 1, 65535 ], "
		       "type = (string) { float32, float64, int32, uint32, int16, uint16, int8, uint8 }, "
		       "framerate = (fraction) [ 0/1, 2147483647/1 ]")
    );

/**
 * @brief The capabilities of the outputs
 *
 * @TODO I'm not sure if the range is to be 1, 65535 or larger
 *
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("other/tensor, "
                       "rank = (int) [ 1, 65535 ], "
                       "dim1 = (int) [ 1, 65535 ], "
                       "dim2 = (int) [ 1, 65535 ], "
                       "dim3 = (int) [ 1, 65535 ], "
                       "dim4 = (int) [ 1, 65535 ], "
		       "type = (string) { float32, float64, int32, uint32, int16, uint16, int8, uint8 }, "
		       "framerate = (fraction) [ 0/1, 2147483647/1 ]")
    );

#define gst_tensor_filter_parent_class parent_class
G_DEFINE_TYPE (GstTensor_Filter, gst_tensor_filter, GST_TYPE_BASE_TRANSFORM);

static void gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_tensor_filter_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf);
static GstFlowReturn gst_tensor_filter_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf);
static GstCaps* gst_tensor_filter_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps,
						  GstCaps *filter);
static GstCaps* gst_tensor_filter_fixate_caps(GstBaseTransform *trans,
                                               GstPadDirection direction,
					       GstCaps *caps,
					       GstCaps *othercaps);
static gboolean gst_tensor_filter_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps,
					    GstCaps *outcaps);
/* GObject vmethod implementations */

/* initialize the tensor_filter's class */
static void
gst_tensor_filter_class_init (GstTensor_FilterClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  GstTensor_FilterClass *klass;

  klass = (GstTensor_FilterClass *) g_class;
  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_filter_set_property;
  gobject_class->get_property = gst_tensor_filter_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_FRAMEWORK,
      g_param_spec_string ("framework", "Framework", "Neural network framework ?",
          "", G_PARAM_READABLE));
  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model filepath", "Filepath to the model file ?",
          "", G_PARAM_READABLE));
  g_object_class_install_property (gobject_class, PROP_INPUT,
      g_param_spec_value_array ("input", "Input dimension", "Input tensor dimension from inner array, upto 4 dimensions ?",
          g_param_spec_uint("dimension", "Dimension", "Size of a dimension axis in input dimension array", 1, 65535, 1, G_PARAM_READABLE), G_PARAM_READABLE));
  g_object_class_install_property (gobject_class, PROP_OUTPUT,
      g_param_spec_value_array ("output", "Output dimension", "Output tensor dimension from inner array, upto 4 dimensions ?",
          g_param_spec_uint("dimension", "Dimension", "Size of a dimension axis in output dimension array", 1, 65535, 1, G_PARAM_READABLE), G_PARAM_READABLE));

  gst_element_class_set_details_simple(gstelement_class,
    "Tensor_Filter",
    "NN Frameworks (e.g., tensorflow) as Media Filters",
    "Handles NN Frameworks (e.g., tensorflow) as Media Filters with other/tensor type stream",
    "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR(gst_tensor_filter_transform);
  trans_class->transform_ip = GST_DEBUG_FUNCPTR(gst_tensor_filter_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps = GST_DEBUG_FUNCPTR(gst_tensor_filter_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR(gst_tensor_filter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR(gst_tensor_filter_set_caps);

  /* Allocation units */
  // @TODO Fill these in!
  // trans_class-> ...
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_filter_init (GstTensor_Filter * filter)
{
  filter->silent = FALSE;
  filter->nnfw = _T_F_UNDEFINED;
  filter->inputConfigured = FALSE;
  filter->outputConfigured = FALSE;
  filter->modelFilename = NULL;

  filter->inputDimension[0] = 1; // innermost
  filter->inputDimension[1] = 1;
  filter->inputDimension[2] = 1;
  filter->inputDimension[3] = 1; // out

  filter->outputDimension[0] = 1; // innermost
  filter->outputDimension[1] = 1;
  filter->outputDimension[2] = 1;
  filter->outputDimension[3] = 1; // out
}

static void
gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_FRAMEWORK:
    case PROP_MODEL:
    case PROP_INPUT:
    case PROP_OUTPUT:
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_FRAMEWORK:
      g_value_set_string(value, nnfw_names[filter->nnfw]);
      break;
    case PROP_MODEL:
      g_value_set_string(value, filter->modelFilename);
      break;
    case PROP_INPUT: {
        GArray *input = g_array_sized_new(FALSE, FALSE, 4, NNS_TENSOR_RANK_LIMIT);
	int i;
	for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
	  g_array_append_val(input, filter->inputDimension[i]);
        g_value_take_boxed(value, input);
	// take function hands the object over from here so that we don't need to free it.
      }
    case PROP_OUTPUT: {
        GArray *output = g_array_sized_new(FALSE, FALSE, 4, NNS_TENSOR_RANK_LIMIT);
	int i;
	for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
	  g_array_append_val(output, filter->outputDimension[i]);
        g_value_take_boxed(value, output);
	// take function hands the object over from here so that we don't need to free it.
      }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/******************************************************************
 * GstElement vmethod implementations
 */

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
tensor_filter_init (GstPlugin * tensor_filter)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template tensor_filter' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_filter_debug, "tensor_filter",
      0, "Template tensor_filter");

  return gst_element_register (tensor_filter, "tensor_filter", GST_RANK_NONE,
      GST_TYPE_TENSOR_FILTER);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_filter"
#endif

/* gstreamer looks for this structure to register tensor_filters
 *
 * exchange the string 'Template tensor_filter' with your tensor_filter description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_filter,
    "tensor_filter",
    tensor_filter_init,
    VERSION,
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)

static GstFlowReturn gst_tensor_filter_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf)
{
  return GST_FLOW_ERROR;
}

static GstFlowReturn gst_tensor_filter_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf)
{
  return GST_FLOW_ERROR;
}

/**
 * gst_tensor_filter_transform_caps() - configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap
 * @filter this element's cap (don't know specifically.)
 */
static GstCaps* gst_tensor_filter_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps,
						  GstCaps *filter)
{
  return NULL;
}

static GstCaps* gst_tensor_filter_fixate_caps(GstBaseTransform *trans,
                                               GstPadDirection direction,
					       GstCaps *caps,
					       GstCaps *othercaps)
{
  return NULL;
}

static gboolean gst_tensor_filter_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps,
					    GstCaps *outcaps)
{
  return FALSE;
}
