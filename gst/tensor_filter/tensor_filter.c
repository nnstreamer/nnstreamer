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
 * in each framework specific files; e.g., tensor_filter_tensorflow_lite.c
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
 * gst-launch -v -m fakesrc ! tensor_filter framework=tensorflow-lite, model=./inception_v3.pb, input=3:224:224, output=1000 ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 *
 * If input is other/tensor C array input[1][224][224][3] and
 * output is other/tensor C array output[1][1][1][1000]
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <glib.h>

#include "tensor_filter.h"

GstTensor_Filter_Framework *tensor_filter_supported[] = {
  [_T_F_UNDEFINED] = NULL,

  [_T_F_CUSTOM] = &NNS_support_custom,
  [_T_F_TENSORFLOW_LITE] = &NNS_support_tensorflow_lite,
  [_T_F_TENSORFLOW] = NULL,
  [_T_F_CAFFE2] = NULL,

  0,
};

const char *nnfw_names[] = {
  [_T_F_UNDEFINED] = "Not supported",

  [_T_F_CUSTOM] = "custom",
  [_T_F_TENSORFLOW_LITE] = "tensorflow-lite",
  [_T_F_TENSORFLOW] = "tensorflow",
  [_T_F_CAFFE2] = "caffe2",

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
  PROP_INPUTTYPE,
  PROP_OUTPUT,
  PROP_OUTPUTTYPE,
  PROP_CUSTOM,
};

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

#define gst_tensor_filter_parent_class parent_class
G_DEFINE_TYPE (GstTensor_Filter, gst_tensor_filter, GST_TYPE_BASE_TRANSFORM);

static void gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_tensor_filter_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static GstCaps *gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
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
      g_param_spec_string ("framework", "Framework",
          "Neural network framework ?", "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model filepath",
          "Filepath to the model file ?", "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_INPUT,
      g_param_spec_string ("input", "Input dimension",
          "Input tensor dimension from inner array, upto 4 dimensions ?", "",
          G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_INPUTTYPE,
      g_param_spec_string ("inputtype", "Input tensor element type",
          "Type of each element of the input tensor ?", "uint8",
          G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_OUTPUT,
      g_param_spec_string ("output", "Output dimension",
          "Output tensor dimension from inner array, upto 4 dimensions ?", "",
          G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_OUTPUTTYPE,
      g_param_spec_string ("outputtype", "Output tensor element type",
          "Type of each element of the output tensor ?", "uint8",
          G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_CUSTOM,
      g_param_spec_string ("custom", "Custom properties for subplugins",
          "Custom properties for subplugins ?", "", G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
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
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_filter_transform);
  trans_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_set_caps);

  /* Allocation units */
  /* @TODO Fill these in!  trans_class-> ... */
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_filter_init (GstTensor_Filter * filter)
{
  filter->prop.silent = TRUE;
  filter->prop.nnfw = _T_F_UNDEFINED;
  filter->prop.fw = NULL;
  filter->prop.inputConfigured = FALSE;
  filter->prop.outputConfigured = FALSE;
  filter->prop.modelFilename = NULL;

  filter->prop.inputDimension[0] = 1;   /* innermost */
  filter->prop.inputDimension[1] = 1;
  filter->prop.inputDimension[2] = 1;
  filter->prop.inputDimension[3] = 1;   /* out */
  filter->prop.inputType = _NNS_END;    /* not initialized */
  filter->prop.inputCapNegotiated = FALSE;

  filter->prop.outputDimension[0] = 1;  /* innermost */
  filter->prop.outputDimension[1] = 1;
  filter->prop.outputDimension[2] = 1;
  filter->prop.outputDimension[3] = 1;  /* out */
  filter->prop.outputType = _NNS_END;   /* not initialized */
  filter->prop.outputCapNegotiated = FALSE;

  filter->prop.customProperties = NULL;
  filter->privateData = NULL;   /* mark not initialized. */
}

/**
 * @brief Calculate the rank of a tensor
 * @param dimension The dimension vector (tensor_dim = uint32_t[NNS_TENSOR_RANK_LIMIT]) of tensor.
 * @return the rank value
 */
static int
gst_tensor_filter_get_rank (tensor_dim dimension)
{
  int i = 0;
  int rank = 0;
  g_assert (dimension);
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    g_assert (dimension[i] > 0);
    if (dimension[i] > 1)
      rank = i + 1;
  }
  if (rank == 0)                // a scalar (assume it is 1-dim vector)
    return 1;
  return rank;
}

/**
 * @brief Fix CAPS for sink/src pad based on input/output metadata in filter.
 * @param filter "this" object
 * @param isInput TRUE if it's for input. FALSE if it's for output.
 * @param targetCaps Caps object to be filters. NULL if we don't have ony.
 * @param checkOnly This is to check the capability for debugging only. Don't return or set anything
 * @return Return the new caps. (returns NULL if checkOnly == TRUE)
 *
 * We need both type and dimension to do this.
 * This is supposed to be used by set_properties, restrting pad-caps before attaching input/output elements
 */
static GstCaps *
gst_tensor_filter_fix_caps (GstTensor_Filter * filter, gboolean isInput,
    GstCaps * targetCaps, gboolean checkOnly)
{
  tensor_type *type = NULL;
  uint32_t *dimension;
  GstCaps *tmp = NULL, *tmp2 = NULL;
  int rank;

  if (isInput == TRUE) {
    type = &(filter->prop.inputType);
    dimension = filter->prop.inputDimension;
  } else {
    type = &(filter->prop.outputType);
    dimension = filter->prop.outputDimension;
  }

  /* 2. configure caps based on type & dimension */
  rank = gst_tensor_filter_get_rank (dimension);
  tmp = gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "type", G_TYPE_STRING, tensor_element_typename[*type], "dim1", G_TYPE_INT, dimension[0], "dim2", G_TYPE_INT, dimension[1], "dim3", G_TYPE_INT, dimension[2], "dim4", G_TYPE_INT, dimension[3], "framerate", GST_TYPE_FRACTION, 0, 1,     /* @TODO: support other framerates! */
      NULL);                    /* Framerate is not determined with the given info */
  if (filter->prop.silent == FALSE) {
    gchar *str = gst_caps_to_string (tmp);
    debug_print (TRUE, "Caps(%s) Narrowing to %s\n",
        (isInput == TRUE) ? "input/sink" : "output/src", str);
    g_free (str);
  }

  if (checkOnly == TRUE) {
    gst_caps_unref (tmp);
    return NULL;
  }

  if (targetCaps) {
    gchar *str;
    if (filter->prop.silent == FALSE) {
      str = gst_caps_to_string (targetCaps);
      debug_print (TRUE, "targetCaps: %s\n", str);
      g_free (str);
    }
    tmp2 = gst_caps_intersect_full (targetCaps, tmp, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp);
    tmp = tmp2;
    if (filter->prop.silent == FALSE) {
      str = gst_caps_to_string (tmp);
      debug_print (TRUE, "resultCaps: %s\n", str);
      g_free (str);
    }
  } else {
    if (filter->prop.silent == FALSE) {
      gchar *str = gst_caps_to_string (tmp);
      debug_print (TRUE, "resultCaps w/o targetCaps: %s\n", str);
      g_free (str);
    }
  }

  /* @TODO 3. Check if tmp ( = targetCap \cap tmp(from dim)) is not \null-set. */

  /* @TODO 4. unref the old cap */

  /* @TODO 5. Verify with get_input/output_dimension callbacks! */

  return tmp;                   // @TODO Incorrect. Do "copy"

}

/**
 * @brief Configure tensor metadata from sink caps.
 *
 * This is a direct import from tensor_converter::gst_tensor_converter_configure_tensor.
 * This checks if the sink-pad-cap is consistant with src-pad-cap.
 * This can be done only by looking at input/ouput dimension queries.
 */
static gboolean
gst_tensor_filter_configure_tensor (const GstCaps * caps,
    GstTensor_Filter * filter)
{
  /* @TODO 1: query input/output dims/types */
  /* @TODO 2: verify it with current caps */
  /* @TODO 3: return value assessment */
  return FALSE;
}

static void
gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER (object);

  debug_print (!filter->prop.silent, "Setting property. for Prop %d.\n",
      prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      filter->prop.silent = g_value_get_boolean (value);
      debug_print (!filter->prop.silent, "Debug mode on (silent off)\n");
      break;
    case PROP_FRAMEWORK:
      g_assert (filter->prop.nnfw == _T_F_UNDEFINED && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.nnfw =
          find_key_strv (nnfw_names, g_value_get_string (value));
      debug_print (!filter->prop.silent, "Framework = %s\n",
          g_value_get_string (value));
      g_assert (filter->prop.nnfw != -1);
      g_assert (filter->prop.nnfw != _T_F_UNDEFINED);
      g_assert (nnfw_support_status[filter->prop.nnfw] == TRUE);
      filter->prop.fw = tensor_filter_supported[filter->prop.nnfw];
      g_assert (filter->prop.fw != NULL);
      break;
    case PROP_MODEL:
      g_assert (filter->prop.modelFilename == NULL && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.modelFilename = g_value_dup_string (value);
      debug_print (!filter->prop.silent, "Model = %s\n",
          filter->prop.modelFilename);
      g_assert (g_file_test (filter->prop.modelFilename,
              G_FILE_TEST_IS_REGULAR) == TRUE);
      break;
    case PROP_INPUT:
      g_assert (filter->prop.inputConfigured == FALSE && value);
      /* Once configures, it cannot be changed in runtime */
      {
        int rank = get_tensor_dimension (g_value_get_string (value),
            filter->prop.inputDimension);
        g_assert (rank > 0 && rank <= NNS_TENSOR_RANK_LIMIT);
        filter->prop.inputConfigured = TRUE;
        debug_print (!filter->prop.silent, "Input Prop: %d:%d:%d:%d Rank %d\n",
            filter->prop.inputDimension[0], filter->prop.inputDimension[1],
            filter->prop.inputDimension[2], filter->prop.inputDimension[3],
            rank);
      }
      if (filter->prop.inputType != _NNS_END && filter->prop.silent == FALSE)
        gst_tensor_filter_fix_caps (filter, TRUE, NULL, TRUE);
      break;
    case PROP_OUTPUT:
      g_assert (filter->prop.outputConfigured == FALSE && value);
      /* Once configures, it cannot be changed in runtime */
      {
        int rank = get_tensor_dimension (g_value_get_string (value),
            filter->prop.outputDimension);
        g_assert (rank > 0 && rank <= NNS_TENSOR_RANK_LIMIT);
        filter->prop.outputConfigured = TRUE;
        debug_print (!filter->prop.silent, "Output Prop: %d:%d:%d:%d Rank %d\n",
            filter->prop.outputDimension[0], filter->prop.outputDimension[1],
            filter->prop.outputDimension[2], filter->prop.outputDimension[3],
            rank);
      }

      if (filter->prop.outputType != _NNS_END && filter->prop.silent == FALSE)
        gst_tensor_filter_fix_caps (filter, FALSE, NULL, TRUE);
      break;
    case PROP_INPUTTYPE:
      g_assert (filter->prop.inputType == _NNS_END && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.inputType = get_tensor_type (g_value_get_string (value));
      debug_print (!filter->prop.silent, "Output Type: %s -> %d\n",
          g_value_get_string (value), filter->prop.inputType);
      g_assert (filter->prop.inputType != _NNS_END);
      if (filter->prop.inputConfigured == TRUE && filter->prop.silent == FALSE)
        gst_tensor_filter_fix_caps (filter, TRUE, NULL, TRUE);
      break;
    case PROP_OUTPUTTYPE:
      g_assert (filter->prop.outputType == _NNS_END && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.outputType = get_tensor_type (g_value_get_string (value));
      g_assert (filter->prop.outputType != _NNS_END);
      if (filter->prop.outputConfigured == TRUE && filter->prop.silent == FALSE)
        gst_tensor_filter_fix_caps (filter, FALSE, NULL, TRUE);
      break;
    case PROP_CUSTOM:
      g_assert (filter->prop.customProperties == NULL && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.customProperties = g_value_dup_string (value);
      if (filter->prop.debug == TRUE)
        g_printerr ("Custom Option = %s\n", filter->prop.customProperties);
      break;
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

  debug_print (!filter->prop.silent, "Getting property. for Prop %d.\n",
      prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->prop.silent);
      break;
    case PROP_FRAMEWORK:
      g_value_set_string (value, nnfw_names[filter->prop.nnfw]);
      break;
    case PROP_MODEL:
      g_value_set_string (value, filter->prop.modelFilename);
      break;
    case PROP_INPUT:{
      GArray *input =
          g_array_sized_new (FALSE, FALSE, 4, NNS_TENSOR_RANK_LIMIT);
      int i;
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
        g_array_append_val (input, filter->prop.inputDimension[i]);
      g_value_take_boxed (value, input);
      /* take function hands the object over from here so that we don't need to free it. */
    }
      break;
    case PROP_OUTPUT:{
      GArray *output =
          g_array_sized_new (FALSE, FALSE, 4, NNS_TENSOR_RANK_LIMIT);
      int i;
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
        g_array_append_val (output, filter->prop.outputDimension[i]);
      g_value_take_boxed (value, output);
      /* take function hands the object over from here so that we don't need to free it. */
    }
      break;
    case PROP_INPUTTYPE:
      g_value_set_string (value,
          tensor_element_typename[filter->prop.inputType]);
      break;
    case PROP_OUTPUTTYPE:
      g_value_set_string (value,
          tensor_element_typename[filter->prop.outputType]);
      break;
    case PROP_CUSTOM:
      g_value_set_string (value, filter->prop.customProperties);
      break;
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
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_filter,
    "tensor_filter",
    tensor_filter_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");

static GstFlowReturn
gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  int ret;
  tensor_dim inputDimChk;
  tensor_dim outputDimChk;
  tensor_type inputType, outputType;
  size_t outBufSize;
  uint8_t *inptr, *outptr;
  GstMapInfo inInfo, outInfo;

  if (G_UNLIKELY (filter->prop.inputCapNegotiated == FALSE
          || filter->prop.outputCapNegotiated == FALSE))
    goto unknown_format;
  if (G_UNLIKELY (!filter->prop.fw))
    goto unknown_framework;
  if (G_UNLIKELY (!filter->prop.modelFilename))
    goto unknown_model;
  if (G_UNLIKELY (!filter->prop.fw->invoke_NN))
    goto unknown_invoke;

  /* 0. Check all properties and inbuf size. */
  debug_print (!filter->prop.silent, "Invoking %s with %s model\n",
      filter->prop.fw->name, filter->prop.modelFilename);

  if (filter->prop.fw->getInputDimension) {
    ret = filter->prop.fw->getInputDimension (filter, inputDimChk, &inputType);
    /* @TODO check inputDimChk / inputType with filter internal info */
  } else {
    /* @TODO printout debug msg */
  }

  if (filter->prop.fw->getOutputDimension) {
    ret =
        filter->prop.fw->getOutputDimension (filter, outputDimChk, &outputType);
    /* @TODO check outputDimChk / outputType with filter internal info */
  } else {
    /* @TODO printout debug msg */
  }

  /* 1. Allocate outbuf */
  g_assert (outbuf);
  outBufSize = tensor_element_size[filter->prop.inputType] *
      get_tensor_element_count (filter->prop.inputDimension);
  if (gst_buffer_get_size (outbuf) < outBufSize) {
    gst_buffer_set_size (outbuf, outBufSize);
  }
  g_assert (gst_buffer_get_size (outbuf) >= outBufSize);

  /* 2. Call the filter-subplugin callback, "invoke" */
  gst_buffer_map (inbuf, &inInfo, GST_MAP_READ);
  gst_buffer_map (outbuf, &outInfo, GST_MAP_WRITE);
  inptr = inInfo.data;
  outptr = outInfo.data;

  ret = filter->prop.fw->invoke_NN (filter, inptr, outptr);

  gst_buffer_unmap (inbuf, &inInfo);
  gst_buffer_unmap (outbuf, &outInfo);

  /* 3. Return result! */
  if (ret)
    return GST_FLOW_ERROR;
  return GST_FLOW_OK;
unknown_format:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL), ("unknown format"));
  return GST_FLOW_NOT_NEGOTIATED;
unknown_framework:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("framework not configured"));
  return GST_FLOW_ERROR;
unknown_model:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("model filepath not configured"));
  return GST_FLOW_ERROR;
unknown_invoke:
  GST_ELEMENT_ERROR (filter, CORE, NOT_IMPLEMENTED, (NULL),
      ("invoke function is not defined"));
  return GST_FLOW_ERROR;
}

static GstFlowReturn
gst_tensor_filter_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  /* @TODO 0. Check all properties and inbuf size. */
  /* @TODO 0-1. This shouldn't reach here if in-place mode if OFF with the subplugin */
  /* @TODO 0-1. , which could be done at *_caps with gst_base_transform_set_in_place() */
  /* @TODO 1. Resize buf if output is larger than input */
  /* @TODO 2. Call the filter-subplugin callback, "invoke" */
  /* @TODO 3. Return result! */
  g_assert (1 == 0);
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
static GstCaps *
gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensor_Filter *obj = GST_TENSOR_FILTER_CAST (trans);

  if (direction == GST_PAD_SINK) {
    /* caps: sink pad. get src pad info */
    obj->prop.outputCapNegotiated = TRUE;

    /* @TODO 1. Check caps w/ getInputDimension && saved input dimension */
    /* @TODO 2. Check returning-caps w/ getOutputDimension && saved output dimension */
    return gst_tensor_filter_fix_caps (obj, FALSE, filter, FALSE);
  } else {
    /* caps: src pad. get sink pad info */
    obj->prop.inputCapNegotiated = TRUE;

    /* @TODO 1. Check caps w/ getOutputDimension && saved output dimension */
    /* @TODO 2. Check returning-caps w/ getInputDimension && saved input dimension */
    return gst_tensor_filter_fix_caps (obj, TRUE, filter, FALSE);
  }

  /* @TODO Cannot reach here. Remove them later */
  g_assert (1 == 0);
  gst_tensor_filter_configure_tensor (caps, obj);
  return NULL;
}

static GstCaps *
gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstCaps *supposed =
      gst_tensor_filter_transform_caps (trans, direction, caps, NULL);
  GstCaps *result = gst_caps_intersect (othercaps, supposed);
  GstTensor_Filter *obj = GST_TENSOR_FILTER_CAST (trans);

  g_assert (!gst_caps_is_empty (result));
  gst_caps_unref (othercaps);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset (caps, result)) {
      gst_caps_replace (&result, caps);
    }
    obj->prop.inputCapNegotiated = TRUE;
  } else {
    obj->prop.outputCapNegotiated = TRUE;
  }
  return result;
}

static gboolean
gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);

  /* @TODO This code is for testing only. Not even with proper design / concepts */
  gst_tensor_filter_configure_tensor (incaps, filter);  /* we will need to supply outcaps as well */
  /* Nothing to do yet. */

  return TRUE;
}
