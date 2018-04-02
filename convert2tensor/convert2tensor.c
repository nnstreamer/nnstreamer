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
    GST_STATIC_CAPS ("video/x-raw, format = (string)RGB, views = (int)1, interlace-mode = (string)progressive")
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
                       "rank = (int) [ 1, 4 ], "
                       "dim1 = (int) [ 1, 65535 ], "
                       "dim2 = (int) [ 1, 65535 ], "
                       "dim3 = (int) [ 1, 65535 ], "
                       "dim4 = (int) [ 1, 65535 ], "
		       "type = (string) { float32, float64, int32, uint32, int16, uint16, int8, uint8 }, "
		       "framerate = (fraction) [ 0/1, 2147483647/1 ]")
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

  /** Allocation units
   *  transform_size and get_unit_size are omitted because we do not change
   *  the size of buffer or unit with the current version.
   */
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
  filter->negotiated = FALSE;
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
  gsize tensorFrameSize;
  gboolean ret;
  GstCaps *outcaps;
  int i;

  /* This caps is coming from video/x-raw */
  structure = gst_caps_get_structure(caps, 0);
  rank = 3; /* [color-space][height][width] */

  return_false_if_fail(gst_structure_get_int(structure, "width", &dimension[1]));
  return_false_if_fail(gst_structure_get_int(structure, "height", &dimension[2]));
  return_false_if_fail(gst_structure_get_fraction(structure, "framerate", &framerate_numerator, &framerate_denominator));
  type = _C2T_UINT8; /* Assume color depth per component is 8 bit */
  if (dimension[1] % 4) {
    g_print("  Width(dim2) is not divisible with 4. Width is adjusted %d -> %d\n",
        dimension[1], (dimension[1] + 3) / 4 * 4);
    dimension[1] = (dimension[1] + 3) / 4 * 4;
  }
  dimension[0] = 3; /* R G B */
  dimension[3] = 1; /* This is 3-D Tensor */
  tensorFrameSize = GstConvert2TensorDataSize[type] * dimension[0] * dimension[1] * dimension[2] * dimension[3];
  /* Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html */

  if (filter->tensorConfigured == TRUE) {
    /* It has been already configured. Check if they are consistent */
    if (rank == filter->rank &&
	type == filter->type &&
	framerate_numerator == filter->framerate_numerator &&
	tensorFrameSize == filter->tensorFrameSize &&
	framerate_denominator == filter->framerate_denominator) {
      for (i = 0; i < GST_CONVERT2TENSOR_TENSOR_RANK_LIMIT; i++)
        if (dimension[i] != filter->dimension[i]) {
	  g_printerr("  Dimension %d Mismatch with cached: %d --> %d\n", i, dimension[i], filter->dimension[i]);
	  return FALSE;
	}
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
  filter->tensorFrameSize = tensorFrameSize;

  filter->tensorConfigured = TRUE;

  /* @TODO Support other types */
  filter->input_media_type = _C2T_VIDEO;
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
#define PACKAGE "convert2tensor"
#endif

/* gstreamer looks for this structure to register convert2tensors
 *
 * exchange the string 'Template convert2tensor' with your convert2tensor description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    convert2tensor,
    "convert2tensor",
    convert2tensor_init,
    VERSION,
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)

static GstFlowReturn gst_c2t_transformer_videoframe(GstConvert2Tensor *filter,
                                               GstVideoFrame *inframe, GstBuffer *outbuf)
{
  return gst_buffer_copy_into(outbuf, inframe->buffer,
      GST_BUFFER_COPY_FLAGS | GST_BUFFER_COPY_TIMESTAMPS, 0,
      GST_VIDEO_FRAME_SIZE(inframe));
}

static GstFlowReturn gst_convert2tensor_transform(GstBaseTransform *trans,
                                                  GstBuffer *inbuf,
                                                  GstBuffer *outbuf)
{
  GstVideoFrame in_frame;
  GstFlowReturn res;
  GstConvert2Tensor *filter = GST_CONVERT2TENSOR_CAST(trans);

  if (G_UNLIKELY(!filter->negotiated))
    goto unknown_format;
  if (G_UNLIKELY(!filter->tensorConfigured))
    goto unknown_tensor;

  switch(filter->input_media_type) {
  case _C2T_VIDEO:
    // CAUTION! in_info.video must be already configured!
    if (!gst_video_frame_map(&in_frame, &filter->in_info.video, inbuf,
            GST_MAP_READ | GST_VIDEO_FRAME_MAP_FLAG_NO_REF))
      goto invalid_buffer;

    if (gst_c2t_transformer_videoframe(filter, &in_frame, outbuf))
      res = GST_FLOW_OK;
    else
      res = GST_FLOW_ERROR;
    gst_video_frame_unmap(&in_frame);
    break;
  /* NOT SUPPORTED */
  case _C2T_AUDIO:
  case _C2T_STRING:
  default:
    g_printerr("  Unsupported Media Type (%d)\n", filter->input_media_type);
    goto unknown_type;
  }

  return res;

unknown_format:
  GST_ELEMENT_ERROR(filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR(filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR(filter, CORE, NOT_IMPLEMENTED, (NULL), ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
invalid_buffer:
  GST_ELEMENT_ERROR(filter, CORE, NOT_IMPLEMENTED, (NULL), ("invalid video buffer received from input"));
  return GST_FLOW_ERROR;
}

static GstFlowReturn gst_convert2tensor_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf)
{
  /* DO NOTHING. THIS WORKS AS A PASSTHROUGH. We just remove metadata from video */
  return GST_FLOW_OK;
}

/**
 * gst_convert2tensor_transform_caps() - configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap
 * @filter this element's cap (don't know specifically.)
 */
static GstCaps* gst_convert2tensor_transform_caps(GstBaseTransform *trans,
                                                  GstPadDirection direction,
						  GstCaps *caps,
						  GstCaps *filter)
{
  GstCaps *tmp;
  gboolean ret;
  GstConvert2Tensor bogusFilter = {0};
  bogusFilter.tensorConfigured = FALSE;

  /* @TODO: Verify if direction == GST_PAD_SINK means caps is sink pad */
  if (direction == GST_PAD_SINK) {
    /* Skip verifying if caps is compatible: let's assume sink_factory will do that. */
    /* @TODO: Verify if this assumption is correct */

    /* Construct bogusFilter from caps (sinkpad) */
    ret = gst_convert2tensor_configure_tensor(caps, &bogusFilter);
    if (ret == FALSE) {
      g_printerr("  Cannot retrieve tensor spec from the given input cap.\n");
      tmp = gst_caps_new_empty();
      return tmp; /* Empty Cap */
    }

    g_assert(bogusFilter.tensorConfigured == TRUE);

    /* Construct GstCap (srcpad) from bugusFilter */
    tmp = gst_caps_new_simple("other/tensor",
        "rank", G_TYPE_INT, bogusFilter.rank,
        "dim1", G_TYPE_INT, bogusFilter.dimension[0],
        "dim2", G_TYPE_INT, bogusFilter.dimension[1],
        "dim3", G_TYPE_INT, bogusFilter.dimension[2],
        "dim4", G_TYPE_INT, bogusFilter.dimension[3],
        "type", G_TYPE_STRING, GstConvert2TensorDataTypeName[bogusFilter.type],
	"framerate", GST_TYPE_FRACTION, bogusFilter.framerate_numerator,
	             bogusFilter.framerate_denominator,
        NULL);

    GST_DEBUG_OBJECT(trans, "transformed %" GST_PTR_FORMAT " into %"
        GST_PTR_FORMAT, caps, tmp);
    return tmp;
  } else if (direction == GST_PAD_SRC) {
    /* Construct possible GstCap (sinkpad) with src_factory */
    /* @TODO This supports video only! */
    GstStaticCaps staticcap =
        GST_STATIC_CAPS("video/x-raw, format = (string)RGB, view = (int)1, "
        "interlace-mode = (string)progressive, "
	"framerate = (fraction) [ 0/1, 2147483647/1 ], "
	"width = (int) [1, 65535], "
	"height = (int) [1, 65535]");
    tmp = gst_static_caps_get(&staticcap);

    GST_DEBUG_OBJECT(trans, "transformed %" GST_PTR_FORMAT " into %"
        GST_PTR_FORMAT, caps, tmp);
    return tmp;
  }
  /* Neither SRC/SINK? Impossible! */
  g_printerr("Direction = %d\n", direction);
  GST_DEBUG_OBJECT(trans, "Error pad direction type. direction: %d", direction);
  g_assert(TRUE == FALSE);
  return NULL;
}

static GstCaps* gst_convert2tensor_fixate_caps(GstBaseTransform *trans,
                                               GstPadDirection direction,
					       GstCaps *caps,
					       GstCaps *othercaps)
{
  GstCaps *supposed = gst_convert2tensor_transform_caps(trans, direction, caps, NULL);
  GstCaps *result;

  GST_DEBUG_OBJECT (trans, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  result = gst_caps_intersect(othercaps, supposed);
  if (gst_caps_is_empty(result)) {
    gst_caps_unref(result);
    result = othercaps;
  } else {
    gst_caps_unref(othercaps);
  }

  GST_DEBUG_OBJECT (trans, "now fixating %" GST_PTR_FORMAT, result);

  result = gst_caps_make_writable(result);
  result = gst_caps_fixate(result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset(caps, result)) {
      gst_caps_replace(&result, caps);
    }
  }
  return result;
}

static gboolean gst_convert2tensor_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps,
					    GstCaps *outcaps)
{
  /** This is notifier of cap changes for subclass.
   *  However, we do not have subclass (This is the concrete class)
   */
  GstConvert2Tensor *filter = GST_CONVERT2TENSOR_CAST(trans);
  GstVideoInfo in_info, out_info;

  GST_DEBUG_OBJECT (trans, "converting from  %" GST_PTR_FORMAT
      " to %" GST_PTR_FORMAT, incaps, outcaps);

  /* @TODO Supports video only */
  /* input caps */
  if (!gst_video_info_from_caps (&in_info, incaps)) {
    g_printerr("Cannot set_caps\n");
    return FALSE;
  }

  filter->in_info.video = in_info;
  gst_base_transform_set_in_place(trans, TRUE);

  filter->negotiated = gst_convert2tensor_configure_tensor(incaps, filter);

  /* @TODO Verity if outcaps and filter conf are compatible */

}
