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
#include <string.h>

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
  filter->prop.fwOpened = FALSE;
  filter->prop.fwClosed = FALSE;
  filter->prop.inputConfigured = _TFC_INIT;
  filter->prop.outputConfigured = _TFC_INIT;
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
 * @brief Invoke callbacks of filter->prop.fw. Gurantees calling open for the first call.
 */
#define gst_tensor_filter_call(filter, funcname, ...) ({ \
    int __ret = 0; \
    do { \
      if (filter->prop.fwOpened == FALSE) { \
        if (filter->prop.fw->open != NULL) \
          filter->prop.fw->open(filter, &filter->privateData); \
	filter->prop.fwOpened = TRUE; \
      } \
      g_assert(filter->prop.fwClosed != TRUE); \
      __ret = filter->prop.fw->funcname(filter, &filter->privateData, __VA_ARGS__); \
    } while(0); \
    __ret; \
})

/* @TODO Call this where appropriate */
#define gst_tensor_filter_close(filter) \
    do { \
      g_assert(filter->prop.fwClosed != TRUE); \
      g_assert(filter->prop.fwOpened == TRUE); \
      if (filter->prop.fw->close) \
        filter->prop.fw->close(filter, &filter->privateData); \
      filter->prop.fw->fwClosed = TRUE; \
    } while (0);

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
  GstTensor_Filter_CheckStatus configured = FALSE;
  GstCaps *tmp = NULL, *tmp2 = NULL;
  int rank;

  if (isInput == TRUE) {
    type = &(filter->prop.inputType);
    dimension = filter->prop.inputDimension;
    configured = filter->prop.inputConfigured & _TFC_ALL;
  } else {
    type = &(filter->prop.outputType);
    dimension = filter->prop.outputDimension;
    configured = filter->prop.outputConfigured & _TFC_ALL;
  }

  /* @TODO KNOWN BUG: when prop.i/o-dim is not configured, this is going to screw all */
  /* This known bug breaks case 4 of nnstreamer_filter_custom */

  /* 2. configure caps based on type & dimension */
  if (configured == _TFC_ALL) {
    /* static cap can be configured. all figured out already. */
    rank = gst_tensor_filter_get_rank (dimension);
    tmp = gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "type", G_TYPE_STRING, tensor_element_typename[*type], "dim1", G_TYPE_INT, dimension[0], "dim2", G_TYPE_INT, dimension[1], "dim3", G_TYPE_INT, dimension[2], "dim4", G_TYPE_INT, dimension[3], "framerate", GST_TYPE_FRACTION, 0, 1,   /* @TODO: support other framerates! */
        NULL);                  /* Framerate is not determined with the given info */
  } else if (configured == _TFC_DIMENSION) {
    /* dimension is set. only the type is not configured (@TODO not sure if this is possible) */
    rank = gst_tensor_filter_get_rank (dimension);
    tmp = gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "dim1", G_TYPE_INT, dimension[0], "dim2", G_TYPE_INT, dimension[1], "dim3", G_TYPE_INT, dimension[2], "dim4", G_TYPE_INT, dimension[3], "framerate", GST_TYPE_FRACTION, 0, 1,  /* @TODO: support other framerates! */
        NULL);                  /* Framerate is not determined with the given info */
  } else if (configured == _TFC_TYPE) {
    /* type is set. only the dim is not configured (@TODO not sure if this is possible) */
    rank = gst_tensor_filter_get_rank (dimension);
    tmp = gst_caps_new_simple ("other/tensor", "framerate", GST_TYPE_FRACTION, 0, 1,    /* @TODO: support other framerates! */
        "type", G_TYPE_STRING, tensor_element_typename[*type], NULL);   /* Framerate is not determined with the given info */
  } else {
    /* knows nothing. This happens.. */
    tmp = gst_caps_new_simple ("other/tensor", "framerate", GST_TYPE_FRACTION, 0, 1,    /* @TODO: support other framerates! */
        NULL);                  /* Framerate is not determined with the given info */
  }

  if (filter->prop.silent == FALSE || configured != _TFC_ALL) {
    gchar *str = gst_caps_to_string (tmp);
    debug_print (TRUE, "Caps(%s) Narrowing from %s\n",
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
 * @brief Configure tensor metadata from sink caps. (internal static function)
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

static inline gboolean
compare_dimension (const tensor_dim Ad, const tensor_dim Bd)
{
  int i;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
    if (Ad[i] != Bd[i])
      return FALSE;
  return TRUE;
}

/**
 * @brief Check consistency between filter->dim & getInput/OutputDimension of fw. (internal static function)
 * @param filter "this" pointer to tensor_filter object.
 * @param checkInput TRUE to check input dimension
 * @param checkOutput TRUE to cehck output dimension
 * @return TRUE if consistent or unknown. FALSE if incosistency is found
 */
static gboolean
gst_tensor_filter_check_consistency_fw (GstTensor_Filter * filter,
    gboolean checkInput, gboolean checkOutput)
{
  GstTensor_Filter_Framework *fw = filter->prop.fw;
  tensor_type type;
  tensor_dim dim;
  int ret;

  if (fw == NULL)
    return TRUE;                /* Nothing to check. FW is not configured, yet */

  if (checkInput == TRUE && fw->getInputDimension != NULL) {
    ret = gst_tensor_filter_call (filter, getInputDimension, dim, &type);
    if (ret) {
      debug_print (TRUE,
          "getInputDimension failed (%d). But we can continue with it.\n", ret);
      /* Cannot get input dimenson. Cannot say "inconsistant"! Don't check values */
    } else {
      if ((filter->prop.inputConfigured & _TFC_DIMENSION) &&
          FALSE == compare_dimension (dim, filter->prop.inputDimension))
        return FALSE;
      if ((filter->prop.inputConfigured & _TFC_TYPE) &&
          type != filter->prop.inputType)
        return FALSE;
    }
  }

  if (checkOutput == TRUE && fw->getOutputDimension != NULL) {
    ret = gst_tensor_filter_call (filter, getOutputDimension, dim, &type);
    if (ret) {
      debug_print (TRUE,
          "getOutputDimension failed (%d). But we can continue with it.\n",
          ret);
      /* Cannot get output dimenson. Cannot say "inconsistant"! Don't check values */
    } else {
      if ((filter->prop.outputConfigured & _TFC_DIMENSION) &&
          FALSE == compare_dimension (dim, filter->prop.outputDimension))
        return FALSE;
      if ((filter->prop.outputConfigured & _TFC_TYPE) &&
          type != filter->prop.outputType)
        return FALSE;
    }
  }

  return TRUE;
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

      /* See if mandatory methods are filled in */
      g_assert (filter->prop.fw->invoke_NN);
      /**
       * Do not check if setInputDim XOR (getInputDim && getOutputDim)
       * at this stage. A subplugin may have all the three callbacks
       * and disable one or two later after the model is loaded.
       * Thus, check setInputDim OR (getInputDim && getOutputDim) here.
       */
      g_assert ((filter->prop.fw->getInputDimension
              && filter->prop.fw->getOutputDimension)
          || filter->prop.fw->setInputDimension);
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
      g_assert (!(filter->prop.inputConfigured & _TFC_DIMENSION) && value);
      /* Once configures, it cannot be changed in runtime */
      {
        int rank = get_tensor_dimension (g_value_get_string (value),
            filter->prop.inputDimension);
        g_assert (rank > 0 && rank <= NNS_TENSOR_RANK_LIMIT);
        filter->prop.inputConfigured |= _TFC_DIMENSION;
        debug_print (!filter->prop.silent, "Input Prop: %d:%d:%d:%d Rank %d\n",
            filter->prop.inputDimension[0], filter->prop.inputDimension[1],
            filter->prop.inputDimension[2], filter->prop.inputDimension[3],
            rank);
      }
      break;
    case PROP_OUTPUT:
      g_assert (!(filter->prop.outputConfigured & _TFC_DIMENSION) && value);
      /* Once configures, it cannot be changed in runtime */
      {
        int rank = get_tensor_dimension (g_value_get_string (value),
            filter->prop.outputDimension);
        g_assert (rank > 0 && rank <= NNS_TENSOR_RANK_LIMIT);
        filter->prop.outputConfigured |= _TFC_DIMENSION;
        debug_print (!filter->prop.silent, "Output Prop: %d:%d:%d:%d Rank %d\n",
            filter->prop.outputDimension[0], filter->prop.outputDimension[1],
            filter->prop.outputDimension[2], filter->prop.outputDimension[3],
            rank);
      }
      break;
    case PROP_INPUTTYPE:
      g_assert (filter->prop.inputType == _NNS_END && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.inputType = get_tensor_type (g_value_get_string (value));
      filter->prop.inputConfigured |= _TFC_TYPE;
      g_assert (filter->prop.inputType != _NNS_END);
      break;
    case PROP_OUTPUTTYPE:
      g_assert (filter->prop.outputType == _NNS_END && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.outputType = get_tensor_type (g_value_get_string (value));
      filter->prop.outputConfigured |= _TFC_TYPE;
      g_assert (filter->prop.outputType != _NNS_END);
      break;
    case PROP_CUSTOM:
      g_assert (filter->prop.customProperties == NULL && value);
      /* Once configures, it cannot be changed in runtime */
      filter->prop.customProperties = g_value_dup_string (value);
      if (filter->prop.silent == FALSE)
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

  g_assert ((filter->prop.inputConfigured & _TFC_ALL) == _TFC_ALL &&
      (filter->prop.outputConfigured & _TFC_ALL) == _TFC_ALL);

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

  ret = gst_tensor_filter_call (filter, invoke_NN, inptr, outptr);

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
 * @brief process property values, call get/set I/O dim. (internal static function)
 * If set-prop configured dimension, verify the dimension with fw callbacks
 * Otherwise, configure dimension with fw callbacks.
 *
 * @param filter "this" pointer
 * @return 1: OK and all set. 0: Try again later. -1: cannot proceed. fatal ERROR.
 */
static int
gst_tensor_filter_property_process (GstTensor_Filter * filter)
{
  GstTensor_Filter_Framework *fw = filter->prop.fw;
  GstTensor_Filter_Properties *prop = &filter->prop;
  int ret;
  tensor_dim dim;
  tensor_type type;
  int i;

  /* Ensure the subplugin is contacted first before checking the XOR assert */
  if (!prop->fwOpened && fw->open)
    fw->open (filter, &filter->privateData);
  prop->fwOpened = TRUE;

  g_assert (!(fw->getInputDimension && fw->getOutputDimension) != !fw->setInputDimension);      /* This is "XOR" */
  if (fw->getInputDimension != NULL) {
    g_assert (fw->getOutputDimension != NULL);

    ret = gst_tensor_filter_call (filter, getInputDimension, dim, &type);
    if (ret) {
      err_print ("getInputDimension() returns %d. Cannot proceed.\n", ret);
      return -1;
    }

    if (!(prop->inputConfigured & _TFC_TYPE)) { /* input type not configured */
      prop->inputType = type;
      prop->inputConfigured |= _TFC_TYPE;
    }
    if (!(prop->inputConfigured & _TFC_DIMENSION)) {    /* input dim not configred */
      memcpy (prop->inputDimension, dim, sizeof (dim));
      prop->inputConfigured |= _TFC_DIMENSION;
    }

    ret = gst_tensor_filter_call (filter, getOutputDimension, dim, &type);
    if (ret) {
      err_print ("getOutputDimension() returns %d. Cannot proceed.\n", ret);
      return -1;
    }

    if (!(prop->outputConfigured & _TFC_TYPE)) {        /* output type not configured */
      prop->outputType = type;
      prop->outputConfigured |= _TFC_TYPE;
    }
    if (!(prop->outputConfigured & _TFC_DIMENSION)) {   /* output dim not configured */
      memcpy (prop->outputDimension, dim, sizeof (dim));
      prop->outputConfigured |= _TFC_DIMENSION;
    }

    if (TRUE == gst_tensor_filter_check_consistency_fw (filter, TRUE, TRUE))
      return 1;
    else
      return -1;

  } else {
    g_assert (fw->getOutputDimension == NULL && fw->setInputDimension != NULL);

    /* If filter's inputdimension is not clear, yet, we cannot proceed. try again later */
    if ((prop->inputConfigured & _TFC_ALL) != _TFC_ALL)
      return 0;

    ret =
        fw->setInputDimension (filter, &filter->privateData,
        prop->inputDimension, prop->inputType, dim, &type);
    if (prop->outputConfigured & _TFC_TYPE) {
      g_assert (prop->outputType == type);
    }
    if (prop->outputConfigured & _TFC_DIMENSION)
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
        g_assert (prop->outputDimension[i] == dim[i]);
    prop->outputType = type;
    memcpy (prop->outputDimension, dim, sizeof (dim));
    prop->outputConfigured |= _TFC_ALL;

    return 1;
  }

  return -1;                    /* Code cannot reach here */
}


/**
 * gst_tensor_filter_transform_caps() - configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap (if direction GST_PAD_SINK)
 * @filter this element's cap (don't know specifically.)
 *
 * Be careful not to fix/set caps at this stage. Negotiation not completed yet.
 */
static GstCaps *
gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensor_Filter *obj = GST_TENSOR_FILTER_CAST (trans);
  int check = gst_tensor_filter_property_process (obj);

  g_assert (check >= 0);

  if (direction == GST_PAD_SINK) {
    /* caps: sink pad. get src pad info */
    obj->prop.outputCapNegotiated = TRUE;

    /* @TODO 1. Check caps w/ getInputDimension && saved input dimension */
    /* @TODO 2. Check returning-caps w/ getOutputDimension && saved output dimension */
    return gst_tensor_filter_fix_caps (obj, FALSE, caps, FALSE);
  } else {
    /* caps: src pad. get sink pad info */
    obj->prop.inputCapNegotiated = TRUE;

    /* @TODO 1. Check caps w/ getOutputDimension && saved output dimension */
    /* @TODO 2. Check returning-caps w/ getInputDimension && saved input dimension */
    return gst_tensor_filter_fix_caps (obj, TRUE, caps, FALSE);
  }

  /* @TODO Cannot reach here. Remove them later */
  g_assert (1 == 0);
  gst_tensor_filter_configure_tensor (caps, obj);
  return NULL;
}

/**
 * @brief Try to generate dim/type from caps (internal static function)
 * @return _TFC_TYPE is on if type determined. _TFC_DIMENSION is on if dim determined
 * @param filter "this" pointer
 * @param caps the caps to be analyzed (padcap)
 * @param[out] dim tensor dimension derived from caps
 * @param[out] type tensor type derived from caps
 */
static GstTensor_Filter_CheckStatus
gst_tensor_filter_generate_dim_from_cap (GstCaps * caps, tensor_dim dim,
    tensor_type * type)
{
  unsigned int i, capsize;
  const GstStructure *str;
  GstTensor_Filter_CheckStatus ret = _TFC_INIT;
  int rank;
  const gchar *strval;

  if (!caps) {
    return _TFC_INIT;
  }

  capsize = gst_caps_get_size (caps);

  for (i = 0; i < capsize; i++) {
    str = gst_caps_get_structure (caps, i);
    if (gst_structure_get_int (str, "dim1", (int *) &dim[0]) &&
        gst_structure_get_int (str, "dim2", (int *) &dim[1]) &&
        gst_structure_get_int (str, "dim3", (int *) &dim[2]) &&
        gst_structure_get_int (str, "dim4", (int *) &dim[3])) {
      int j;
      ret |= _TFC_DIMENSION;
      if (gst_structure_get_int (str, "rank", &rank)) {
        for (j = rank; j < NNS_TENSOR_RANK_LIMIT; j++)
          g_assert (dim[j] == 1);
      }
    }
    strval = gst_structure_get_string (str, "type");
    if (strval) {
      *type = get_tensor_type (strval);
      g_assert (*type != _NNS_END);
      ret |= _TFC_TYPE;
    }
  }

  return ret;
}

/**
 * @brief Read pad-cap and return dimension/type info
 * @return _TFC_TYPE is on if type determined. _TFC_DIMENSION is on if dim determined
 * @param[in] caps The pad cap
 * @param[in] input TRUE if input. FALSE if output.
 * @param[out] dim Tensor dimension
 @ @param[out[ type Tensor element type
 */
static void
gst_tensor_caps_to_dimension (GstCaps * caps, gboolean input,
    GstTensor_Filter_Properties * prop)
{
  if (input) {
    prop->inputConfigured |=
        gst_tensor_filter_generate_dim_from_cap (caps, prop->inputDimension,
        &prop->inputType);
  } else {
    prop->outputConfigured |=
        gst_tensor_filter_generate_dim_from_cap (caps, prop->outputDimension,
        &prop->outputType);
  }
}

static GstCaps *
gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstCaps *supposed =
      gst_tensor_filter_transform_caps (trans, direction, caps, NULL);
  GstCaps *result = gst_caps_intersect (othercaps, supposed);
  GstTensor_Filter *obj = GST_TENSOR_FILTER_CAST (trans);
  GstTensor_Filter_Framework *fw = obj->prop.fw;
  GstCaps *sinkpadcap, *srcpadcap;
  int check = gst_tensor_filter_property_process (obj);

  g_assert (check >= 0);

  g_assert (!gst_caps_is_empty (result));
  gst_caps_unref (othercaps);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  if (direction == GST_PAD_SINK) {
    if (gst_caps_is_subset (caps, result)) {
      gst_caps_replace (&result, caps);
    }
    obj->prop.inputCapNegotiated = TRUE;
    sinkpadcap = caps;
    srcpadcap = result;
  } else {
    obj->prop.outputCapNegotiated = TRUE;
    sinkpadcap = result;
    srcpadcap = caps;
  }

  if ((obj->prop.inputConfigured & _TFC_ALL) == _TFC_ALL &&
      (obj->prop.outputConfigured & _TFC_ALL) == _TFC_ALL)
    return result;

  debug_print (!obj->prop.silent, "Nego (%s) / i %d / o %d\n",
      (direction == GST_PAD_SINK) ? "sink" : "src",
      obj->prop.inputCapNegotiated, obj->prop.outputCapNegotiated);

  /* Before moving on, use if getInputDim/getOutputDim is available. */
  if (fw->getInputDimension
      && (obj->prop.inputConfigured & _TFC_ALL) == _TFC_ALL) {
    int ret = gst_tensor_filter_call (obj, getInputDimension,
        obj->prop.inputDimension, &obj->prop.inputType);
    if (ret == 0) {
      obj->prop.inputConfigured |= _TFC_ALL;
    }
  }
  if (fw->getOutputDimension
      && (obj->prop.outputConfigured & _TFC_ALL) == _TFC_ALL) {
    int ret = gst_tensor_filter_call (obj, getOutputDimension,
        obj->prop.outputDimension, &obj->prop.outputType);
    if (ret == 0) {
      obj->prop.outputConfigured |= _TFC_ALL;
    }
  }
  if ((obj->prop.inputConfigured & _TFC_ALL) == _TFC_ALL &&
      (obj->prop.outputConfigured & _TFC_ALL) == _TFC_ALL) {
    return result;
  }

  gst_tensor_caps_to_dimension (sinkpadcap, TRUE, &obj->prop);
  gst_tensor_caps_to_dimension (srcpadcap, FALSE, &obj->prop);

  if ((obj->prop.inputConfigured & _TFC_ALL) == _TFC_ALL &&
      (obj->prop.outputConfigured & _TFC_ALL) == _TFC_ALL)
    return result;

  if ((obj->prop.inputConfigured & _TFC_ALL) == _TFC_ALL) {
    if (fw->setInputDimension) {
      int ret = gst_tensor_filter_call (obj, setInputDimension,
          obj->prop.inputDimension, obj->prop.inputType,
          obj->prop.outputDimension, &obj->prop.outputType);
      obj->prop.outputConfigured |= _TFC_ALL;
      g_assert (ret == 0);
      return result;
    }
  }

  /**
   * @TODO ARCH-Decision required; are we going to (and do we need to)
   * support setOutputDimention (and get InputDim accordingly?)
   *
   * If not, we have done with it and emit error here if we still don't have
   * capabilities fixed.
   *
   * In this case, result should be re-calculated because
   * gst_tensor_filter_transform_caps () cannot do reverse transform.
   */

  if (!obj->prop.silent) {
    gchar *str = gst_caps_to_string (caps);
    debug_print (TRUE, "Caps(%s) %s\n",
        (direction == GST_PAD_SINK) ? "input/sink" : "output/src", str);
    g_free (str);
    str = gst_caps_to_string (result);
    debug_print (TRUE, "Caps(%s) %s\n",
        (direction == GST_PAD_SINK) ? "Op-input/sink" : "Op-output/src", str);
    g_free (str);
  }

  g_assert (0);                 /* Not Supported (configure input from output dimension) */
  return result;
}

static gboolean
gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  int check = gst_tensor_filter_property_process (filter);
  tensor_dim dim;
  tensor_type type;
  gboolean result;

  g_assert (check >= 0);

  /* @TODO This code is for testing only. Not even with proper design / concepts */
  gst_tensor_filter_configure_tensor (incaps, filter);  /* we will need to supply outcaps as well */
  /* Nothing to do yet. */

  result = gst_tensor_filter_generate_dim_from_cap (incaps, dim, &type);
  /* @TODO Configure filter-dim from caps if filter-dim is not configured, yet */
  if ((filter->prop.inputConfigured & _TFC_ALL) != _TFC_ALL) {
    /* we may set if result == TRUE */
    g_assert (FALSE);           /* NYI */

    g_assert (result == TRUE);
  }
  /* @TODO Check consistencyu between dim/type with filter->input* */

  result = gst_tensor_filter_generate_dim_from_cap (outcaps, dim, &type);
  /* @TODO Configure filter-dim from caps if filter-dim is not configured, yet */
  if ((filter->prop.outputConfigured & _TFC_ALL) != _TFC_ALL) {
    /* we may set if result == TRUE */
    g_assert (FALSE);           /* NYI */

    g_assert (result == TRUE);
  }
  /* @TODO Check consistencyu between dim/type with filter->output* */

  return TRUE;
}
