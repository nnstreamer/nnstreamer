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
 *
 * @file	tensordec.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert tensors (as a filter for other general neural network filters) to other media types
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 *
 */

/**
 * SECTION:element-tensordec
 *
 * A filter that converts tensor stream for NN frameworks to media stream.
 * The input is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesink ! tensordec ! fakesrc silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>
#include <glib/gprintf.h>

#include "tensordec.h"
#ifdef TIZEN
#include <dlog.h>
#else
#define dlog_print(...) do {} while(0)
#endif

GST_DEBUG_CATEGORY_STATIC (gst_tensordec_debug);
#define GST_CAT_DEFAULT gst_tensordec_debug

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

/**
 * @brief The capabilities of the inputs
 *
 * In v0.0.1, this is 3-d tensor, [color][height][width]
 *
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

/* the capabilities of the outputs
 *
 * In v0.0.1, this is "bitmap" image stream
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

#define gst_tensordec_parent_class parent_class
G_DEFINE_TYPE (GstTensorDec, gst_tensordec, GST_TYPE_BASE_TRANSFORM);

static void gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_tensordec_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static GstCaps *gst_tensordec_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensordec_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensordec_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
/* GObject vmethod implementations */

/**
 * @breif initialize the tensordec's class
 */
static void
gst_tensordec_class_init (GstTensorDecClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  GstTensorDecClass *klass;

  klass = (GstTensorDecClass *) g_class;
  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensordec_set_property;
  gobject_class->get_property = gst_tensordec_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorDec",
      "Convert tensor stream to media stream ",
      "Converts tensor stream of C-Array for neural network framework filters to audio or video stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensordec_transform);
  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_tensordec_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensordec_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensordec_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensordec_set_caps);

  /** Allocation units
   *  @TODO Need to add Allocation units depending on the tensor size.
   */
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensordec_init (GstTensorDec * filter)
{
  filter->silent = FALSE;
  filter->Configured = FALSE;
  filter->negotiated = FALSE;
  filter->addPadding = FALSE;
}

static void
gst_tensordec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDec *filter = GST_TENSORDEC (object);

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
gst_tensordec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDec *filter = GST_TENSORDEC (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* @brief Return 1 if we need to add stride per row from the stream data */
static int
add_stride_padding_per_row (const gchar * format, int width)
{
  /* @TODO The actual list is much longer. fill them (read https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html ) */
  if ((!g_strcmp0 (format, "RGB") || !g_strcmp0 (format, "BGR")
          || !g_strcmp0 (format, "I420")) && (width % 4))
    return 1;
  return 0;
}


/******************************************************************
 * GstElement vmethod implementations
 */

#define return_false_if_fail(x)                 \
  ret = (x);                                    \
  if (!ret)                                     \
    return FALSE;                               \
  ;
/**
 * @brief Configure tensor metadata from sink caps
 */
static gboolean
gst_tensordec_configure (const GstCaps * caps, GstTensorDec * filter)
{
  GstStructure *structure;
  tensor_type type;
  gint framerate_numerator;
  gint framerate_denominator;
  gint width;
  gint height;
  gboolean ret;
  gchar format[1024];
  gchar interlace[1024];
  const gchar *type_str;
  int i;

  /* This caps is coming from tensor */
  structure = gst_caps_get_structure (caps, 0);
  return_false_if_fail (gst_structure_get_int (structure, "dim1", &i));
  /* @TODO Need to support orther media format (RGB, BRG, YUV,.. etc.). And should support Audio as well.*/
  filter->output_media_type = _NNS_VIDEO;
  if (i == 3 || i == 4) {
    filter->format = i;
    if (i == 3)
      strcpy (format, "RGB");
    else
      strcpy (format, "BGRx");
  } else {
    return FALSE;
  }

  return_false_if_fail (gst_structure_get_int (structure, "dim2", &width));
  return_false_if_fail (gst_structure_get_int (structure, "dim3", &height));

  return_false_if_fail (gst_structure_get_fraction (structure, "framerate",
          &framerate_numerator, &framerate_denominator));

  type_str = gst_structure_get_string (structure, "type");
  if (!g_strcmp0 (type_str, "uint8")) {
    type = _NNS_UINT8;
  } else {
    return FALSE;
  }


  /* Emit Warning if RSTRIDE = RU4 (3BPP) && Width % 4 > 0 */
  /* @TODO: Add more conditions! */
  if (add_stride_padding_per_row (format, width)) {
    g_print
        ("  Width(dim2) is not divisible with 4. The performance won't be good; one more memcpy is added.\n");
    dlog_print (DLOG_WARN, "nnstreamer",
        "Input video width is not divisible with 4. The performance will not be good.");
    filter->addPadding = TRUE;
  }

  if (filter->Configured == TRUE) {
    /* It has been already configured. Check if they are consistent */
    if (width == filter->width &&
        height == filter->height &&
        type == filter->type &&
        framerate_numerator == filter->framerate_numerator &&
        framerate_denominator == filter->framerate_denominator) {
      return TRUE;
    }
    g_printerr ("  Something's wrong. The tensor metadata is inconsistent.\n");
    return FALSE;
  }

  filter->type = type;
  filter->framerate_numerator = framerate_numerator;
  filter->framerate_denominator = framerate_denominator;
  filter->width = width;
  filter->height = height;

  filter->Configured = TRUE;

  /* @TODO Need to specify of video mode*/
  filter->views = 1;
  strcpy (interlace, "progressive");
  if (!g_strcmp0 (interlace, "progressive")) {
    filter->mode = _VIDEO_PROGRESSIVE;
  } else {
    /* dose not support others for now */
    return FALSE;
  }

  /* @TODO Support other types */
  filter->output_media_type = _NNS_VIDEO;
  return TRUE;
}


/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
tensordec_init (GstPlugin * tensordec)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template tensordec' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensordec_debug, "tensordec",
      0, "Template tensordec");

  return gst_element_register (tensordec, "tensordec", GST_RANK_NONE,
      GST_TYPE_TENSORDEC);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensordec"
#endif

/* gstreamer looks for this structure to register tensordecs
 *
 * exchange the string 'Template tensordec' with your tensordec description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensordec,
    "tensordec",
    tensordec_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/")

     static GstFlowReturn gst_tensordec_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorDec *filter = GST_TENSORDEC_CAST (trans);

  if (G_UNLIKELY (!filter->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!filter->Configured))
    goto unknown_tensor;

  switch (filter->output_media_type) {
    case _NNS_VIDEO:
      break;
      /* NOT SUPPORTED */
    case _NNS_AUDIO:
    case _NNS_STRING:
    default:
      g_printerr ("  Unsupported Media Type (%d)\n", filter->output_media_type);
      goto unknown_type;
  }

  return GST_FLOW_OK;

unknown_format:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
}

static GstFlowReturn
gst_tensordec_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstTensorDec *filter = GST_TENSORDEC_CAST (trans);

  if (G_UNLIKELY (!filter->negotiated))
    goto unknown_format;
  if (G_UNLIKELY (!filter->Configured))
    goto unknown_tensor;

  switch (filter->output_media_type) {
    case _NNS_VIDEO:
      if (filter->addPadding == TRUE) {
        /* @TODO Add Padding for x-raw */
      }
      break;
      /* NOT SUPPORTED */
    case _NNS_AUDIO:
    case _NNS_STRING:
    default:
      g_printerr ("  Unsupported Media Type (%d)\n", filter->output_media_type);
      goto unknown_type;
  }

  /* DO NOTHING. THIS WORKS AS A PASSTHROUGH. We just remove metadata from video */
  return GST_FLOW_OK;

unknown_format:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_tensor:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("unknown format for tensor"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_type:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("not implemented type of media"));
  return GST_FLOW_NOT_SUPPORTED;
}

/**
 * @brief configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap
 * @filter this element's cap (don't know specifically.)
 */
static GstCaps *
gst_tensordec_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstCaps *tmp;
  gboolean ret;
  GstTensorDec bogusFilter = { 0 };
  bogusFilter.Configured = FALSE;

  /* if direction is sink, check src (depending on sink's format, we could choose video or autiod. But currently only video/x-raw (RGB and RGBx) support. cap is for tensor. */
  if (direction == GST_PAD_SINK) {
    GstStructure *structure;
    gchar *str;

    ret = gst_tensordec_configure (caps, &bogusFilter);

    if (ret == FALSE) {
      GstStructure *structure = gst_caps_get_structure (caps, 0);
      gchar *str = gst_structure_to_string (structure);
      gchar str2[2048];
      gchar framerate[1024], width[1024], height[1024], colors[1024];
      /* const gchar *format; */
      int fn = -1, fd, w, h, c;

      if (TRUE == gst_structure_get_fraction (structure, "framerate", &fn, &fd))
        g_sprintf (framerate, "%d/%d", fn, fd);
      else
        g_sprintf (framerate, "[ 0/1, 2147483647/1 ]");

      if (TRUE == gst_structure_get_int (structure, "dim2", &w))
        g_sprintf (width, "%d", w);
      else
        g_sprintf (width, "[1, 65535]");

      if (TRUE == gst_structure_get_int (structure, "dim3", &h))
        g_sprintf (height, "%d", h);
      else
        g_sprintf (height, "[1, 65535]");

      if (TRUE == gst_structure_get_int (structure, "dim1", &c)) {
        if (c == 3)
          g_sprintf (colors, "%s", "RGB");
        else if (c == 4)
          g_sprintf (colors, "%s", "RGBx");
      } else {
        g_sprintf (colors, "{RGB, BGRx}");
      }

      g_printerr ("Structure from caps = %s\n", str);

      g_sprintf (str2,
          "video/x-raw, "
          "format = (string)%s, "
          "views = (int)1, "
          "interlace-mode = (string)progressive, "
          "framerate = (fraction) %s, "
          "width = (int) %s, "
          "height = (int) %s", colors, framerate, width, height);

      tmp = gst_caps_from_string (str2);
      g_printerr ("Structure from caps to = %s\n", str2);

      /* If given caps are in range for width/height,
         we cannot configure tensor, however, we may return proper srcpad caps */
      /* @TODO: see if the error is from ranging width/height before entering here */
      return tmp;

      g_printerr ("  Cannot retrieve tensor spec from the given input cap.\n");
      tmp = gst_caps_new_empty ();
      return tmp;               /* Empty Cap */
    }
    g_printerr ("transform_caps SINK specific\n");

    g_assert (bogusFilter.Configured == TRUE);

    /* @TODO Need to support other format */
    gchar color[1024], interlace[1024];
    if (bogusFilter.format == 3)
      g_sprintf (color, "%s", "RGB");
    else if (bogusFilter.format == 4)
      g_sprintf (color, "%s", "BGRx");
    else
      g_sprintf (color, "%s", "{RGB, BGRx}");
    if (bogusFilter.mode == _VIDEO_PROGRESSIVE)
      g_sprintf (interlace, "progressive");

    tmp = gst_caps_new_simple ("video/x-raw",
        "width", G_TYPE_INT, bogusFilter.width,
        "height", G_TYPE_INT, bogusFilter.height,
        "format", G_TYPE_STRING, color,
        "views", G_TYPE_INT, bogusFilter.views,
        "interlace-mode", G_TYPE_STRING, interlace,
        "framerate", GST_TYPE_FRACTION, bogusFilter.framerate_numerator,
        bogusFilter.framerate_denominator, NULL);

    if (filter) {
      GstCaps *tmp2 =
          gst_caps_intersect_full (filter, tmp, GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (tmp);
      tmp = tmp2;
    }
    structure = gst_caps_get_structure (caps, 0);
    str = gst_structure_to_string (structure);
    g_printerr ("From = %s\n", str);
    structure = gst_caps_get_structure (tmp, 0);
    str = gst_structure_to_string (structure);
    g_printerr ("To = %s\n", str);

    GST_DEBUG_OBJECT (trans, "SINK transformed %" GST_PTR_FORMAT " into %"
        GST_PTR_FORMAT, caps, tmp);
    return tmp;
  } else if (direction == GST_PAD_SRC) {

    GstStructure *structure;
    gchar *str;

    /* Construct possible GstCap (sinkpad) with src_factory */
    /* @TODO This supports video only! */
    GstStaticCaps staticcap = GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT);

    tmp = gst_static_caps_get (&staticcap);

    /* Because SRC is ANY, I'm not sure of this check. */
    if (!gst_caps_is_any (caps)) {
      structure = gst_caps_get_structure (caps, 0);
      str = gst_structure_to_string (structure);
      g_printerr ("Structure from src = %s\n", str);
      if (filter) {
        GstCaps *tmp2;
        structure = gst_caps_get_structure (filter, 0);
        str = gst_structure_to_string (structure);
        g_printerr ("Structure from filter = %s\n", str);

        tmp2 = gst_caps_intersect_full (filter, tmp, GST_CAPS_INTERSECT_FIRST);

        structure = gst_caps_get_structure (tmp2, 0);
        str = gst_structure_to_string (structure);
        g_printerr ("Structure from intersection = %s\n", str);

        gst_caps_unref (tmp);
        tmp = tmp2;
      }

      GST_DEBUG_OBJECT (trans, "SRC transformed %" GST_PTR_FORMAT " into %"
          GST_PTR_FORMAT, caps, tmp);
    }
    return tmp;
  }
  /* Neither SRC/SINK? Impossible! */
  g_printerr ("Direction = %d\n", direction);
  GST_DEBUG_OBJECT (trans, "Error pad direction type. direction: %d",
      direction);
  g_assert (TRUE == FALSE);
  return NULL;
}

static GstCaps *
gst_tensordec_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstCaps *supposed =
      gst_tensordec_transform_caps (trans, direction, caps, NULL);
  GstCaps *result;

  GST_DEBUG_OBJECT (trans, "trying to fixate othercaps %" GST_PTR_FORMAT
      " based on caps %" GST_PTR_FORMAT, othercaps, caps);

  result = gst_caps_intersect (othercaps, supposed);
  if (gst_caps_is_empty (result)) {
    gst_caps_unref (result);
    result = othercaps;
  } else {
    gst_caps_unref (othercaps);
  }

  GST_DEBUG_OBJECT (trans, "now fixating %" GST_PTR_FORMAT, result);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset (caps, result)) {
      gst_caps_replace (&result, caps);
    }
  }
  return result;
}

static gboolean
gst_tensordec_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{

  GstTensorDec *filter = GST_TENSORDEC_CAST (trans);
  gboolean AddPadding = TRUE;

  if (filter->addPadding)
    AddPadding = FALSE;
  gst_base_transform_set_in_place (trans, AddPadding);

  filter->negotiated = gst_tensordec_configure (incaps, filter);

  return TRUE;
}
