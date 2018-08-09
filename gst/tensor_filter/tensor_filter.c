/**
 * GStreamer Tensor_Filter
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 */
/**
 * @file	tensor_filter.c
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
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

#ifdef DISABLE_TENSORFLOW_LITE
  [_T_F_TENSORFLOW_LITE] = NULL,
#else
  [_T_F_TENSORFLOW_LITE] = &NNS_support_tensorflow_lite,
#endif
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
static gboolean gst_tensor_filter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
static gboolean gst_tensor_filter_start (GstBaseTransform * trans);
static gboolean gst_tensor_filter_stop (GstBaseTransform * trans);
/* GObject vmethod implementations */

/**
 * @brief initialize the tensor_filter's class
 */
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
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_size);

  /* start/stop to call open/close */
  trans_class->start = GST_DEBUG_FUNCPTR (gst_tensor_filter_start);
  trans_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_filter_stop);
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 * @todo change the first index [0] of input/output Dimension & Type to loop for multi tensors
 */
static void
gst_tensor_filter_init (GstTensor_Filter * filter)
{
  GstTensor_Filter_Properties *prop = &filter->prop;

  prop->silent = TRUE;
  prop->nnfw = _T_F_UNDEFINED;
  prop->fw = NULL;
  prop->fwOpened = FALSE;
  prop->fwClosed = FALSE;
  prop->inputConfigured = _TFC_INIT;
  prop->outputConfigured = _TFC_INIT;
  prop->modelFilename = NULL;

  prop->inputDimension[0][0] = 1;       /* innermost */
  prop->inputDimension[0][1] = 1;
  prop->inputDimension[0][2] = 1;
  prop->inputDimension[0][3] = 1;       /* out */
  prop->inputType[0] = _NNS_END;        /* not initialized */
  prop->inputCapNegotiated = FALSE;

  prop->outputDimension[0][0] = 1;      /* innermost */
  prop->outputDimension[0][1] = 1;
  prop->outputDimension[0][2] = 1;
  prop->outputDimension[0][3] = 1;      /* out */
  prop->outputType[0] = _NNS_END;       /* not initialized */
  prop->outputCapNegotiated = FALSE;

  prop->customProperties = NULL;
  filter->privateData = NULL;   /* mark not initialized. */
}

#define silent_debug(...) debug_print (!prop->silent, __VA_ARGS__)

/**
 * @brief Invoke callbacks of filter->prop.fw. Gurantees calling open for the first call.
 */
#define gst_tensor_filter_call(filter, ret, funcname, ...) do { \
      if (filter->prop.fwOpened == FALSE) { \
        if (filter->prop.fw->open != NULL) \
          filter->prop.fw->open(filter, &filter->privateData); \
	filter->prop.fwOpened = TRUE; \
      } \
      g_assert(filter->prop.fwClosed != TRUE); \
      ret = filter->prop.fw->funcname(filter, &filter->privateData, __VA_ARGS__); \
    } while(0)

/** @todo Call this where appropriate */
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
  if (rank == 0)                /* a scalar (assume it is 1-dim vector) */
    return 1;
  return rank;
}

static GstTensor_Filter_CheckStatus
gst_tensor_filter_generate_dim_from_cap (GstCaps * caps, tensor_dim dim,
    tensor_type * type);
/**
 * @brief Find caps based on i/o configuration or from the 'other' cap
 * @param filter "this" object
 * @param isInput TRUE if the "source" is input(sinkpad) and "target" is output(srcpad0
 * @param fromCaps The "source" cap given
 * @return The "target" cap from "source" cap.
 *
 * We need both type and dimension to do this.
 * This is supposed to be used by set_properties, restrting pad-caps before attaching input/output elements
 *
 * @todo Looks like this is buggy!!!
 */
static GstCaps *
gst_tensor_filter_fix_caps (GstTensor_Filter * filter, gboolean isInput,
    GstCaps * fromCaps)
{
  tensor_type *type = NULL, _type;
  uint32_t *dimension[NNS_TENSOR_SIZE_LIMIT];
  tensor_dim dim[NNS_TENSOR_SIZE_LIMIT];
  GstTensor_Filter_CheckStatus configured = _TFC_INIT;
  GstTensor_Filter_Properties *prop = &filter->prop;
  GstCaps *tmp = NULL, *tmp2 = NULL, *staticcap = NULL, *resultCaps = NULL;
  GstStaticCaps rawcap = GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT);
  int rank;
  staticcap = gst_static_caps_get (&rawcap);

  if (isInput == TRUE) {
    type = prop->inputType;
    int i;
    for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
      dimension[i] = prop->inputDimension[i];
    }
    configured = prop->inputConfigured & _TFC_ALL;
  } else {
    type = prop->outputType;
    int i;
    for (i = 0; i < NNS_TENSOR_SIZE_LIMIT; i++) {
      dimension[i] = prop->outputDimension[i];
    }
    configured = prop->outputConfigured & _TFC_ALL;
  }

  /* 2. configure caps based on type & dimension */
  if (configured == _TFC_ALL) {
    rank = gst_tensor_filter_get_rank (dimension[0]);
    tmp2 =
        gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "type",
        G_TYPE_STRING, tensor_element_typename[type[0]], "dim1", G_TYPE_INT,
        dimension[0][0], "dim2", G_TYPE_INT, dimension[0][1], "dim3",
        G_TYPE_INT, dimension[0][2], "dim4", G_TYPE_INT, dimension[0][3], NULL);
    tmp = gst_caps_intersect_full (staticcap, tmp2, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp2);
  } else if (configured == _TFC_DIMENSION) {
    rank = gst_tensor_filter_get_rank (dimension[0]);
    tmp2 =
        gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "dim1",
        G_TYPE_INT, dimension[0][0], "dim2", G_TYPE_INT, dimension[0][1],
        "dim3", G_TYPE_INT, dimension[0][2], "dim4", G_TYPE_INT,
        dimension[0][3], NULL);
    tmp = gst_caps_intersect_full (staticcap, tmp2, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp2);
  } else if (configured == _TFC_TYPE) {
    tmp2 =
        gst_caps_new_simple ("other/tensor", "type", G_TYPE_STRING,
        tensor_element_typename[type[0]], NULL);
    tmp = gst_caps_intersect_full (staticcap, tmp2, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp2);
  } else {
    /* knows nothing. This happens.. */
    tmp2 = gst_caps_new_any ();
    tmp = gst_caps_intersect_full (staticcap, tmp2, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp2);
  }

  if (fromCaps) {
    gchar *str;
    if (prop->silent == FALSE) {
      str = gst_caps_to_string (fromCaps);
      debug_print (TRUE, "fromCaps: %s\n", str);
      g_free (str);

      str = gst_caps_to_string (tmp);
      debug_print (TRUE, "filter: %s\n", str);
      g_free (str);
    }
    tmp2 = gst_caps_intersect_full (fromCaps, tmp, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (tmp);
    tmp = tmp2;
    if (prop->silent == FALSE) {
      str = gst_caps_to_string (tmp);
      debug_print (TRUE, "filtered fromCaps: %s\n", str);
      g_free (str);
    }
  } else {
    if (prop->silent == FALSE) {
      gchar *str = gst_caps_to_string (tmp);
      debug_print (TRUE, "not filtered fromCaps: %s\n", str);
      g_free (str);
    }
  }

  /* 2-2. Extract effective dim info from tmp */
  dimension[0] = dim[0];
  configured =
      gst_tensor_filter_generate_dim_from_cap (tmp, dimension[0], &_type);
  configured &= _TFC_ALL;
  /* tmp is no more needed */
  gst_caps_unref (tmp);

  /* 3. Calculate resultcap from fromcap. */
  if (isInput == TRUE) {
    /* result == srcpad (output) */
    tensor_dim rdim;
    tensor_type rtype;
    int ret = -1;

    /* 3-1-1. Try get output dim for srcpad */
    if (prop->fw->getOutputDimension)
      gst_tensor_filter_call (filter, ret, getOutputDimension, rdim, &rtype);
    /* 3-1-1-a. If inputdim is available but outputdim is not available */
    if (ret != 0 && configured == _TFC_ALL && prop->fw->setInputDimension) {
      gst_tensor_filter_call (filter, ret, setInputDimension, dimension[0],
          _type, rdim, &rtype);
    }
    /* if ret == 0, either get or set has been successful. */
    if (ret != 0) {
      /* We do not have enough info for dimension */
      /* knows nothing. This happens.. */
      tmp = gst_caps_new_any ();
      resultCaps =
          gst_caps_intersect_full (staticcap, tmp, GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (tmp);
    }

    /* 3-1.2. Configure resultCap from rdim/rtype */
    if (resultCaps == NULL) {
      rank = gst_tensor_filter_get_rank (rdim);
      resultCaps =
          gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "type",
          G_TYPE_STRING, tensor_element_typename[rtype], "dim1", G_TYPE_INT,
          rdim[0], "dim2", G_TYPE_INT, rdim[1], "dim3", G_TYPE_INT,
          rdim[2], "dim4", G_TYPE_INT, rdim[3], NULL);
    }
  } else {
    /* result == sinkpad (input) */
    tensor_dim rdim;
    tensor_type rtype;
    int ret = -1;

    /* 3-1-1. Try get output dim for srcpad */
    if (prop->fw->getInputDimension)
      gst_tensor_filter_call (filter, ret, getInputDimension, rdim, &rtype);
    if (ret != 0) {
      /* We do not have output->input dimension conversion. */
      /* knows nothing. This happens.. */
      tmp = gst_caps_new_any ();
      resultCaps =
          gst_caps_intersect_full (staticcap, tmp, GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (tmp);
    }

    /* 3-1.2. Configure resultCap from rdim/rtype */
    if (resultCaps == NULL) {
      rank = gst_tensor_filter_get_rank (rdim);
      resultCaps =
          gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank,
          "type", G_TYPE_STRING, tensor_element_typename[rtype], "dim1",
          G_TYPE_INT, rdim[0], "dim2", G_TYPE_INT, rdim[1], "dim3",
          G_TYPE_INT, rdim[2], "dim4", G_TYPE_INT, rdim[3], NULL);
    }
  }

  /** @todo 5. Verify with get_input/output_dimension callbacks! */
  gst_caps_unref (staticcap);

  return resultCaps;
}

/**
 * @brief @todo fill this in
 */
static void
gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER (object);
  GstTensor_Filter_Properties *prop = &filter->prop;
  GstTensor_Filter_Framework *fw = prop->fw;

  silent_debug ("Setting property. for Prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      prop->silent = g_value_get_boolean (value);
      silent_debug ("Debug mode on (silent off)\n");
      break;
    case PROP_FRAMEWORK:
      g_assert (prop->nnfw == _T_F_UNDEFINED && value);
      /* Once configures, it cannot be changed in runtime */
      prop->nnfw = find_key_strv (nnfw_names, g_value_get_string (value));
      silent_debug ("Framework = %s\n", g_value_get_string (value));
      g_assert (prop->nnfw != -1);
      g_assert (prop->nnfw != _T_F_UNDEFINED);
      g_assert (tensor_filter_supported[prop->nnfw] != NULL);
      prop->fw = tensor_filter_supported[prop->nnfw];
      fw = prop->fw;
      g_assert (prop->fw != NULL);

      /* See if mandatory methods are filled in */
      g_assert (fw->invoke_NN);
      g_assert ((fw->getInputDimension && fw->getOutputDimension)
          || fw->setInputDimension);
      break;
    case PROP_MODEL:
      g_assert (prop->modelFilename == NULL && value);
      /* Once configures, it cannot be changed in runtime */
      prop->modelFilename = g_value_dup_string (value);
      silent_debug ("Model = %s\n", prop->modelFilename);
      g_assert (g_file_test (prop->modelFilename,
              G_FILE_TEST_IS_REGULAR) == TRUE);
      break;
    case PROP_INPUT:
      g_assert (!(prop->inputConfigured & _TFC_DIMENSION) && value);
      /* Once configures, it cannot be changed in runtime */
      {
        int rank = get_tensor_dimension (g_value_get_string (value),
            prop->inputDimension[0]);
        g_assert (rank > 0 && rank <= NNS_TENSOR_RANK_LIMIT);
        prop->inputConfigured |= _TFC_DIMENSION;
        silent_debug ("Input Prop: %d:%d:%d:%d Rank %d\n",
            prop->inputDimension[0][0], prop->inputDimension[0][1],
            prop->inputDimension[0][2], prop->inputDimension[0][3], rank);
      }
      break;
    case PROP_OUTPUT:
      g_assert (!(prop->outputConfigured & _TFC_DIMENSION) && value);
      /* Once configures, it cannot be changed in runtime */
      {
        int rank = get_tensor_dimension (g_value_get_string (value),
            prop->outputDimension[0]);
        g_assert (rank > 0 && rank <= NNS_TENSOR_RANK_LIMIT);
        prop->outputConfigured |= _TFC_DIMENSION;
        silent_debug ("Output Prop: %d:%d:%d:%d Rank %d\n",
            prop->outputDimension[0][0], prop->outputDimension[0][1],
            prop->outputDimension[0][2], prop->outputDimension[0][3], rank);
      }
      break;
    case PROP_INPUTTYPE:
      g_assert (prop->inputType[0] == _NNS_END && value);
      /* Once configures, it cannot be changed in runtime */
      prop->inputType[0] = get_tensor_type (g_value_get_string (value));
      prop->inputConfigured |= _TFC_TYPE;
      g_assert (prop->inputType[0] != _NNS_END);
      break;
    case PROP_OUTPUTTYPE:
      g_assert (prop->outputType[0] == _NNS_END && value);
      /* Once configures, it cannot be changed in runtime */
      prop->outputType[0] = get_tensor_type (g_value_get_string (value));
      prop->outputConfigured |= _TFC_TYPE;
      g_assert (prop->outputType[0] != _NNS_END);
      break;
    case PROP_CUSTOM:
      g_assert (prop->customProperties == NULL && value);
      /* Once configures, it cannot be changed in runtime */
      prop->customProperties = g_value_dup_string (value);
      if (prop->silent == FALSE)
        g_printerr ("Custom Option = %s\n", prop->customProperties);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief @todo fill this in
 */
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
        g_array_append_val (input, filter->prop.inputDimension[0][i]);
      g_value_take_boxed (value, input);
      /* take function hands the object over from here so that we don't need to free it. */
    }
      break;
    case PROP_OUTPUT:{
      GArray *output =
          g_array_sized_new (FALSE, FALSE, 4, NNS_TENSOR_RANK_LIMIT);
      int i;
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
        g_array_append_val (output, filter->prop.outputDimension[0][i]);
      g_value_take_boxed (value, output);
      /* take function hands the object over from here so that we don't need to free it. */
    }
      break;
    case PROP_INPUTTYPE:
      g_value_set_string (value,
          tensor_element_typename[filter->prop.inputType[0]]);
      break;
    case PROP_OUTPUTTYPE:
      g_value_set_string (value,
          tensor_element_typename[filter->prop.outputType[0]]);
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

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
tensor_filter_init (GstPlugin * tensor_filter)
{
  /**
   * debug category for fltering log messages
   *
   * exchange the string 'Template tensor_filter' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_filter_debug, "tensor_filter",
      0, "Template tensor_filter");

  return gst_element_register (tensor_filter, "tensor_filter", GST_RANK_NONE,
      GST_TYPE_TENSOR_FILTER);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_filter"
#endif

/**
 * gstreamer looks for this structure to register tensor_filters
 *
 * exchange the string 'Template tensor_filter' with your tensor_filter description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_filter,
    "tensor_filter",
    tensor_filter_init, VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");

/**
 * @brief @todo fill this in
 */
static GstFlowReturn
gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  size_t outBufSize;
  uint8_t *inptr, *outptr;
  uint8_t *retoutptr;
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

  /* 1. Allocate outbuf if allocate_in_invoke is FALSE */
  g_assert (outbuf);

  if (filter->prop.fw->allocate_in_invoke == FALSE) {
    outBufSize = tensor_element_size[filter->prop.outputType[0]] *
        get_tensor_element_count (filter->prop.outputDimension[0]);
    if (gst_buffer_get_size (outbuf) < outBufSize) {
      /** @todo: write a routine to say aloud when this happens */
      gst_buffer_set_size (outbuf, outBufSize);
    }
    debug_print (!filter->prop.silent, "outbuf = %lu / expected = %lu\n",
        gst_buffer_get_size (outbuf), outBufSize);
    g_assert (gst_buffer_get_size (outbuf) >= outBufSize);

    /* 2. Call the filter-subplugin callback, "invoke" */
    gst_buffer_map (inbuf, &inInfo, GST_MAP_READ);
    gst_buffer_map (outbuf, &outInfo, GST_MAP_WRITE);
    inptr = inInfo.data;
    outptr = outInfo.data;

    gst_tensor_filter_call (filter, retoutptr, invoke_NN, inptr, outptr);
    g_assert (outptr == retoutptr);

    gst_buffer_unmap (inbuf, &inInfo);
    gst_buffer_unmap (outbuf, &outInfo);
  } else {
    GstMemory *mem;
    gst_buffer_map (inbuf, &inInfo, GST_MAP_READ);
    g_assert (gst_buffer_get_size (outbuf) == 0);

    inptr = inInfo.data;
    gst_tensor_filter_call (filter, retoutptr, invoke_NN, inptr, NULL);
    gst_buffer_unmap (inbuf, &inInfo);

    /** @todo Performance: cache get_tensor_element_count * tensor_element_size */
    mem = gst_memory_new_wrapped (0, retoutptr,
        get_tensor_element_count (filter->prop.outputDimension[0]) *
        tensor_element_size[filter->prop.outputType[0]],
        0,
        get_tensor_element_count (filter->prop.outputDimension[0]) *
        tensor_element_size[filter->prop.outputType[0]], NULL, NULL);
    gst_buffer_insert_memory (outbuf, -1, mem);
  }

  /* 3. Return result! */
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

/**
 * @brief @todo fill this in
 */
static GstFlowReturn
gst_tensor_filter_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  /** @todo 0. Check all properties and inbuf size. */
  /** @todo 0-1. This shouldn't reach here if in-place mode if OFF with the subplugin */
  /** @todo 0-1. , which could be done at *_caps with gst_base_transform_set_in_place() */
  /** @todo 1. Resize buf if output is larger than input */
  /** @todo 2. Call the filter-subplugin callback, "invoke" */
  /** @todo 3. Return result! */
  g_assert (1 == 0);
  return GST_FLOW_ERROR;
}


/**
 * @brief process property values, call get/set I/O dim. (internal static function)
 * If set-prop configured dimension, verify the dimension with fw callbacks
 * Otherwise, configure dimension with fw callbacks.
 *
 * @param filter "this" pointer
 * @param fixate TRUE if we may fixate property values.
 * @return 1: OK and all set. 0: Try again later. -1: cannot proceed. fatal ERROR.
 */
static int
gst_tensor_filter_property_process (GstTensor_Filter * filter, gboolean fixate)
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

  if (fw->getInputDimension != NULL) {
    gst_tensor_filter_call (filter, ret, getInputDimension, dim, &type);
    if (ret == 0) {
      if (prop->inputConfigured & _TFC_TYPE)
        if (prop->inputType[0] != type)
          return -1;
      if (prop->inputConfigured & _TFC_DIMENSION)
        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
          if (prop->inputDimension[0][i] != dim[i])
            return -1;
      if (fixate && !(prop->inputConfigured & _TFC_TYPE)) {
        prop->inputType[0] = type;
        prop->inputConfigured |= _TFC_TYPE;
      }
      if (fixate && !(prop->inputConfigured & _TFC_DIMENSION)) {
        memcpy (prop->inputDimension[0], dim, sizeof (dim));
        prop->inputConfigured |= _TFC_DIMENSION;
      }
    }
  }

  if (fw->getOutputDimension != NULL) {
    gst_tensor_filter_call (filter, ret, getOutputDimension, dim, &type);
    if (ret == 0) {
      if (prop->outputConfigured & _TFC_TYPE)
        if (prop->outputType[0] != type)
          return -1;
      if (prop->outputConfigured & _TFC_DIMENSION)
        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++)
          if (prop->outputDimension[0][i] != dim[i])
            return -1;
      if (fixate && !(prop->outputConfigured & _TFC_TYPE)) {
        prop->outputType[0] = type;
        prop->outputConfigured |= _TFC_TYPE;
      }
      if (fixate && !(prop->outputConfigured & _TFC_DIMENSION)) {
        memcpy (prop->outputDimension[0], dim, sizeof (dim));
        prop->outputConfigured |= _TFC_DIMENSION;
      }
    }
  }

  if (fw->setInputDimension != NULL) {
    tensor_dim idim, *cmpdim;
    tensor_type itype, *cmptype;
    /* If filter's inputdimension is not clear, yet, we cannot proceed. try again later */
    if ((prop->inputConfigured & _TFC_ALL) == _TFC_ALL) {
      cmpdim = &(prop->inputDimension[0]);
      cmptype = &(prop->inputType[0]);
    } else {
      if (fw->getInputDimension != NULL) {
        gst_tensor_filter_call (filter, ret, getInputDimension, idim, &itype);
        if (ret != 0)
          goto finalize;
        cmpdim = &idim;
        cmptype = &itype;
      } else {
        /* Nothing to do here */
        goto finalize;
      }
    }

    gst_tensor_filter_call (filter, ret, setInputDimension, *cmpdim, *cmptype,
        dim, &type);
    if (ret != 0)
      goto finalize;

    if (prop->outputConfigured & _TFC_TYPE) {
      if (prop->outputType[0] != type)
        return -1;
    }
    if (prop->outputConfigured & _TFC_DIMENSION) {
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        if (prop->outputDimension[0][i] != dim[i])
          return -1;
      }
    }

    if (fixate && !(prop->outputConfigured & _TFC_TYPE)) {
      prop->outputType[0] = type;
      prop->outputConfigured |= _TFC_TYPE;
    }
    if (fixate && !(prop->outputConfigured & _TFC_DIMENSION)) {
      memcpy (prop->outputDimension[0], dim, sizeof (dim));
      prop->outputConfigured |= _TFC_DIMENSION;
    }
  }

finalize:
  if ((prop->inputConfigured & _TFC_ALL) == _TFC_ALL &&
      (prop->outputConfigured & _TFC_ALL) == _TFC_ALL)
    return 1;
  else
    return 0;

  return -1;                    /* Code cannot reach here */
}


/**
 * @brief configure tensor-srcpad cap from "proposed" cap.
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
  int check = gst_tensor_filter_property_process (obj, FALSE);

  g_assert (check >= 0);

  if (direction == GST_PAD_SINK) {
    /* caps: sink pad. get src pad info */
    obj->prop.outputCapNegotiated = TRUE;

    /** @todo 1. Check caps w/ getInputDimension && saved input dimension */
    /** @todo 2. Check returning-caps w/ getOutputDimension && saved output dimension */

    return gst_tensor_filter_fix_caps (obj, TRUE, caps);
  } else {
    /* caps: src pad. get sink pad info */
    obj->prop.inputCapNegotiated = TRUE;

    /** @todo 1. Check caps w/ getOutputDimension && saved output dimension */
    /** @todo 2. Check returning-caps w/ getInputDimension && saved input dimension */
    return gst_tensor_filter_fix_caps (obj, FALSE, caps);
  }

  /* Cannot reach here. */
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
        gst_tensor_filter_generate_dim_from_cap (caps, prop->inputDimension[0],
        &prop->inputType[0]);
  } else {
    prop->outputConfigured |=
        gst_tensor_filter_generate_dim_from_cap (caps, prop->outputDimension[0],
        &prop->outputType[0]);
  }
}

/**
 * @brief @todo fill this in
 */
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
  int check = gst_tensor_filter_property_process (obj, TRUE);

  gst_caps_unref (supposed);
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
    int ret = 0;
    gst_tensor_filter_call (obj, ret, getInputDimension,
        obj->prop.inputDimension[0], &obj->prop.inputType[0]);
    if (ret == 0) {
      obj->prop.inputConfigured |= _TFC_ALL;
    }
  }
  if (fw->getOutputDimension
      && (obj->prop.outputConfigured & _TFC_ALL) == _TFC_ALL) {
    int ret = 0;
    gst_tensor_filter_call (obj, ret, getOutputDimension,
        obj->prop.outputDimension[0], &obj->prop.outputType[0]);
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
      int ret = 0;
      gst_tensor_filter_call (obj, ret, setInputDimension,
          obj->prop.inputDimension[0], obj->prop.inputType[0],
          obj->prop.outputDimension[0], &obj->prop.outputType[0]);
      obj->prop.outputConfigured |= _TFC_ALL;
      g_assert (ret == 0);
      return result;
    }
  }

  /**
   * @todo ARCH-Decision required; are we going to (and do we need to)
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

/**
 * @brief @todo fill this in
 */
static gboolean
gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  int check = gst_tensor_filter_property_process (filter, TRUE);
  tensor_dim dim;
  tensor_type type;
  gboolean result;

  g_assert (check >= 0);

  result = gst_tensor_filter_generate_dim_from_cap (incaps, dim, &type);
  /** @todo Configure filter-dim from caps if filter-dim is not configured, yet */
  if ((filter->prop.inputConfigured & _TFC_ALL) != _TFC_ALL) {
    /* we may set if result == TRUE */
    g_assert (FALSE);           /* NYI */

    g_assert (result == TRUE);
  }
  /** @todo Check consistencyu between dim/type with filter->input* */

  result = gst_tensor_filter_generate_dim_from_cap (outcaps, dim, &type);
  /** @todo Configure filter-dim from caps if filter-dim is not configured, yet */
  if ((filter->prop.outputConfigured & _TFC_ALL) != _TFC_ALL) {
    /* we may set if result == TRUE */
    g_assert (FALSE);           /* NYI */

    g_assert (result == TRUE);
  }
  /** @todo Check consistencyu between dim/type with filter->output* */

  return TRUE;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * We cannot directly get the value from size value, we need to review the pad-caps.
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_filter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  const GstCaps *srccap = (direction == GST_PAD_SINK) ? othercaps : caps;
  tensor_dim dim;
  tensor_type type;
  GstTensor_Filter_CheckStatus ret =
      get_tensor_from_padcap (srccap, dim, &type, NULL, NULL);

  if (filter->prop.fw->allocate_in_invoke == TRUE) {
    *othersize = 0;             /* Do not allocate outbuf. invoke_NN will allocate! */
    return TRUE;
  }

  g_assert ((ret & _TFC_ALL) == _TFC_ALL);

  if (!filter->prop.silent) {
    debug_print (TRUE, "transform_size, direction = %s\n",
        (direction == GST_PAD_SINK) ? "sink" : "src");
    GstStructure *structure = gst_caps_get_structure (caps, 0);
    gchar *str = gst_structure_to_string (structure);
    debug_print (TRUE, "cap = %s\n", str);
    g_free (str);
    structure = gst_caps_get_structure (othercaps, 0);
    str = gst_structure_to_string (structure);
    debug_print (TRUE, "othercap = %s\n", str);
    g_free (str);
  }

  *othersize = get_tensor_element_count (dim) * tensor_element_size[type];

  return TRUE;
}


/**
 * @brief Called when the element starts processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_start (GstBaseTransform * trans)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  GstTensor_Filter_Framework *fw = filter->prop.fw;
  GstTensor_Filter_Properties *prop = &filter->prop;

  if (!prop->fwOpened && fw->open)
    fw->open (filter, &filter->privateData);
  prop->fwOpened = TRUE;

  g_assert (prop->fwClosed == FALSE);

  return TRUE;
}

/**
 * @brief Called when the element stops processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_stop (GstBaseTransform * trans)
{
  GstTensor_Filter *filter = GST_TENSOR_FILTER_CAST (trans);
  GstTensor_Filter_Framework *fw = filter->prop.fw;
  GstTensor_Filter_Properties *prop = &filter->prop;

  g_assert (prop->fwOpened == TRUE);

  if (fw->close)
    fw->close (filter, &filter->privateData);
  prop->fwClosed = TRUE;

  if (prop->modelFilename) {
    g_free ((void *) prop->modelFilename);
    prop->modelFilename = NULL;
  }

  if (prop->customProperties) {
    g_free ((void *) prop->customProperties);
    prop->customProperties = NULL;
  }

  return TRUE;
}
