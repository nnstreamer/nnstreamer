/**
 * GStreamer
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
 * @file	tensor_transform.c
 * @date	10 Jul 2018
 * @brief	GStreamer plugin to transform other/tensor dimensions
 * @see		http://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		This is NYI.
 *
 */

/**
 * SECTION:element-tensor_transform
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_converter ! tensor_transform mode=dimchg option=0:2 ! fakesink silent=TRUE
 * ]|
 * <title>How to use dimchg</title>
 * |[
 * option=0:2 # Move 0th dim to 2nd dim. I.e., [a][H][W][C] ==> [a][C][H][W]
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>

#include "tensor_transform.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_transform_debug);
#define GST_CAT_DEFAULT gst_tensor_transform_debug

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
  PROP_MODE,
  PROP_OPTION,
};

#define silent_debug(...) debug_print (!filter->silent, __VA_ARGS__)

/**
 * the capabilities of the inputs
 *
 * In v0.0.1, this is "bitmap" image stream
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

#define gst_tensor_transform_parent_class parent_class
G_DEFINE_TYPE (GstTensor_Transform, gst_tensor_transform,
    GST_TYPE_BASE_TRANSFORM);

static void gst_tensor_transform_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_transform_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

/* GstBaseTransformer vmethod implementations */
static GstFlowReturn gst_tensor_transform_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstCaps *gst_tensor_transform_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_transform_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_transform_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_transform_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
/* GObject vmethod implementations */

/**
 * @brief initialize the tensor_transform's class
 */
static void
gst_tensor_transform_class_init (GstTensor_TransformClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;
  GstTensor_TransformClass *klass;

  klass = (GstTensor_TransformClass *) g_class;
  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_transform_set_property;
  gobject_class->get_property = gst_tensor_transform_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_string ("mode", "Mode", "Tensor transform mode ?",
          "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_OPTION,
      g_param_spec_string ("option", "Option",
          "Option for the tensor transform mode ?", "", G_PARAM_READWRITE));

  gst_element_class_set_details_simple (gstelement_class,
      "Tensor_Transform",
      "Transform other/tensor dimensions",
      "Transforms other/tensor dimensions for different models or frameworks",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_transform_transform);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_transform_transform_caps);
  trans_class->fixate_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_transform_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_transform_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_transform_transform_size);
}

/**
 * @brief initialize the new element (G_DEFINE_TYPE requires this)
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_transform_init (GstTensor_Transform * filter)
{
  filter->silent = TRUE;
  filter->mode = GTT_END;
  filter->option = NULL;
  filter->loaded = FALSE;
}

static const gchar *gst_tensor_transform_mode_string[] = {
  [GTT_DIMCHG] = "dimchg",
  [GTT_END] = "error",
};

/**
 * @brief Get the corresponding mode from the string value
 * @param[in] str The string value for the mode
 * @return corresponding mode for the string. GTT_END for errors
 */
static tensor_transform_mode
gst_tensor_transform_get_mode (const gchar * str)
{
  int i;

  for (i = 0; i < GTT_END; i++) {
    if (!g_ascii_strcasecmp (gst_tensor_transform_mode_string[i], str))
      return i;
  }
  return GTT_END;
}

/**
 * @brief Setup internal data (data_* in GstTensor_Transform)
 * @param[in/out] filter "this" pointer. mode & option MUST BE set already.
 */
static void
gst_tensor_transform_set_option_data (GstTensor_Transform * filter)
{
  if (filter->mode == GTT_END || filter->option == NULL)
    return;
  switch (filter->mode) {
    case GTT_DIMCHG:
    {
      int a, b;
      gchar **strv = g_strsplit (filter->option, ":", 2);
      if (strv[0] != NULL)
        a = g_ascii_strtoull (strv[0], NULL, 10);
      else
        a = 0;
      if (strv[1] != NULL)
        b = g_ascii_strtoull (strv[1], NULL, 10);
      else
        b = 0;
      filter->data_dimchg.from = a;
      filter->data_dimchg.to = b;
      filter->loaded = TRUE;
    }
      break;
    default:
      g_printerr ("Cannot identify mode\n");
      g_assert (0);
  }
}

/**
 * @brief Set property (gst element vmethod)
 */
static void
gst_tensor_transform_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensor_Transform *filter = GST_TENSOR_TRANSFORM (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_MODE:
      filter->mode = gst_tensor_transform_get_mode (g_value_get_string (value));
      g_assert (filter->mode != GTT_END);
      silent_debug ("Mode = %d(%s)\n", filter->mode,
          gst_tensor_transform_mode_string[filter->mode]);
      gst_tensor_transform_set_option_data (filter);
      break;
    case PROP_OPTION:
      filter->option = g_value_dup_string (value);
      silent_debug ("Option = %s\n", filter->option);
      gst_tensor_transform_set_option_data (filter);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_transform_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensor_Transform *filter = GST_TENSOR_TRANSFORM (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_MODE:
      g_value_set_string (value,
          gst_tensor_transform_mode_string[filter->mode]);
      break;
    case PROP_OPTION:
      g_value_set_string (value, filter->option);
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

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
tensor_transform_init (GstPlugin * tensor_transform)
{
  /**
   * debug category for fltering log messages
   *
   * exchange the string 'Template tensor_transform' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_transform_debug, "tensor_transform",
      0, "Template tensor_transform");

  return gst_element_register (tensor_transform, "tensor_transform",
      GST_RANK_NONE, GST_TYPE_TENSOR_TRANSFORM);
}

/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensor_transform"
#endif

/**
 * gstreamer looks for this structure to register tensor_transforms
 *
 * exchange the string 'Template tensor_transform' with your tensor_transform description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_transform,
    "tensor_transform",
    tensor_transform_init,
    VERSION, "LGPL", "GStreamer", "http://gstreamer.net/");

/**
 * @brief subrouting for tensor-tranform, "dimchg" case.
 * @param[in/out] filter "this" pointer
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_dimchg (GstTensor_Transform * filter,
    const uint8_t * inptr, uint8_t * outptr)
{
  /** @todo NYI */
  uint32_t *fromDim = filter->fromDim;
  uint32_t *toDim = filter->toDim;
  int from = filter->data_dimchg.from;
  int to = filter->data_dimchg.to;
  int i, j, k;
  unsigned int loopLimit = 1;
  size_t loopBlockSize = tensor_element_size[filter->type];

  size_t copyblocksize = tensor_element_size[filter->type];
  size_t copyblocklimit = 1;


  if (from == to) {
    /** Useless memcpy. Do not call this or @todo do "IP" operation */
    memcpy (outptr, inptr, get_tensor_element_count (filter->fromDim) *
        tensor_element_size[filter->type]);
    g_printerr
        ("Calling tensor_transform with high memcpy overhead WITHOUT any effects! Check your stream wheter you really need tensor_transform.\n");
    return GST_FLOW_OK;
  }

  g_assert (from >= 0 && from < NNS_TENSOR_RANK_LIMIT);
  g_assert (to >= 0 && to < NNS_TENSOR_RANK_LIMIT);
  g_assert (fromDim[from] == toDim[to]);

  if (from < to) {
    /**
     * Smaller-loop-ed a to larger-loop-ed b
     * E.g., [N][H][W][c] (c:W:H:N) --> [N][c][H][W] (W:H:c:N)
     *
     * @todo CRITICAL-TODO: Optimize the performance!
     */
    for (i = NNS_TENSOR_RANK_LIMIT - 1; i > to; i--)
      loopLimit *= toDim[i];
    for (i = 0; i < to; i++)
      loopBlockSize *= toDim[i];

    for (i = 0; i < from; i++)
      copyblocksize *= fromDim[i];
    for (i = 0; i < to; i++)
      copyblocklimit *= toDim[i];

    for (i = 0; i < loopLimit; i++) {
      /* [i1][i2][...][iN][b][...] i = i1 x i2 x ... x iN */
      uint8_t *destptr = outptr + loopBlockSize * toDim[to] * i;
      const uint8_t *srcptr = inptr + loopBlockSize * toDim[to] * i;

      for (j = 0; j < toDim[to]; j++) {
        uint8_t *j_destptr = destptr + loopBlockSize * j;
        for (k = 0; k < copyblocklimit; k++) {
          memcpy (j_destptr + copyblocksize * k,
              srcptr + k * copyblocksize * toDim[to] + j * copyblocksize,
              copyblocksize);
        }
      }
    }
  } else {
    /**
     * Larger-loop-ed a to smaller-loop-ed b
     * E.g., [N][c][H][W] (W:H:c:N) --> [N][H][W][c] (c:W:H:N)
     * @todo NYI
     */
    g_assert (0);
  }

  return GST_FLOW_OK;
}

/**
 * @brief non-ip transform. required vmethod for BaseTransform class.
 * @param[in/out] trans "super" pointer
 * @param[in] inbuf The input gst buffer
 * @param[out] outbuf The output gst buffer
 * @return Gst Flow Status
 */
static GstFlowReturn
gst_tensor_transform_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstFlowReturn res;
  GstTensor_Transform *filter = GST_TENSOR_TRANSFORM_CAST (trans);

  uint8_t *inptr, *outptr;
  GstMapInfo inInfo, outInfo;

  gst_buffer_map (inbuf, &inInfo, GST_MAP_READ);
  gst_buffer_map (outbuf, &outInfo, GST_MAP_WRITE);
  inptr = inInfo.data;
  outptr = outInfo.data;

  switch (filter->mode) {
    case GTT_DIMCHG:
      res = gst_tensor_transform_dimchg (filter, inptr, outptr);
      break;
    default:
      res = GST_FLOW_NOT_SUPPORTED;
  }

  gst_buffer_unmap (inbuf, &inInfo);
  gst_buffer_unmap (outbuf, &outInfo);

  return res;
}

/**
 * @brief Read cap, write dim/type from the cap.
 * @param[in] caps The input caps to be read
 * @param[out] dim The corresponding tensor-dim from caps
 * @param[out] type The corresponding tensor-type from caps
 * @param[out] frate_num Framerate, numerator
 * @param[out] frate_den Framerate, denomincator
 * @return TRUE if successful (both dim/type read). FALSE if not.
 */
static gboolean inline
gst_tensor_read_cap (GstCaps * caps, tensor_dim dim, tensor_type * type,
    gint * frate_num, gint * frate_den)
{
  GstTensor_Filter_CheckStatus ret = get_tensor_from_padcap (caps, dim, type,
      frate_num, frate_den);

  return (ret & (_TFC_ALL | _TFC_FRAMERATE)) == (_TFC_ALL | _TFC_FRAMERATE);
}

/**
 * @brief Calculate the rank of a tensor
 * @param dimension The dimension vector (tensor_dim = uint32_t[NNS_TENSOR_RANK_LIMIT]) of tensor.
 * @return the rank value
 */
static int
get_rank (const tensor_dim dimension)
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

/**
 * @brief Write cap from the given dim/type. You need to free the returned value
 * @param[in] dim The given tensor dimension
 * @param[in] type The given tensor element type
 * @param[in] fn Framerate numerator
 * @param[in] fd Framerate denominator
 * @return The new allocated GstCaps from the given dim/type.
 */
static GstCaps *
gst_tensor_write_cap (const tensor_dim dim, tensor_type type, gint fn, gint fd)
{
  int rank = get_rank (dim);
  GstStaticCaps rawcap = GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT);
  GstCaps *tmp, *tmp2;
  GstCaps *staticcap = gst_static_caps_get (&rawcap);

  if (fn != -1 && fd != -1) {
    tmp2 =
        gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "type",
        G_TYPE_STRING, tensor_element_typename[type], "dim1", G_TYPE_INT,
        dim[0], "dim2", G_TYPE_INT, dim[1], "dim3", G_TYPE_INT,
        dim[2], "dim4", G_TYPE_INT, dim[3],
        "framerate", GST_TYPE_FRACTION, fn, fd, NULL);
  } else {
    tmp2 =
        gst_caps_new_simple ("other/tensor", "rank", G_TYPE_INT, rank, "type",
        G_TYPE_STRING, tensor_element_typename[type], "dim1", G_TYPE_INT,
        dim[0], "dim2", G_TYPE_INT, dim[1], "dim3", G_TYPE_INT,
        dim[2], "dim4", G_TYPE_INT, dim[3], NULL);
  }
  tmp = gst_caps_intersect_full (staticcap, tmp2, GST_CAPS_INTERSECT_FIRST);
  gst_caps_unref (tmp2);

  return tmp;
}

/**
 * @brief Dimension conversion calculation
 * @param[in] filter "this" pointer
 * @param[in] direction GST_PAD_SINK if input->output conv
 * @param[in] srcDim dimension of source tensor (input if direction is SINK)
 * @param[out] destDim dimension of destinatino tensor (output if direction is SINK)
 * @return TRUE if success
 */
static gboolean
gst_tensor_dimension_conversion (GstTensor_Transform * filter,
    GstPadDirection direction, const tensor_dim srcDim, tensor_dim destDim)
{
  gboolean ret = FALSE;
  switch (filter->mode) {
    case GTT_DIMCHG:
      if (direction == GST_PAD_SINK) {
        int i;
        int a = filter->data_dimchg.from;
        int b = filter->data_dimchg.to;

        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          if (i < a && i < b) {
            destDim[i] = srcDim[i];
          } else if (i > a && i > b) {
            destDim[i] = srcDim[i];
          } else if (a > b) {
            if (i == b) {
              destDim[i] = srcDim[a];
            } else {
              g_assert (i > 0 && i > b);
              destDim[i] = srcDim[i - 1];
            }
          } else if (a < b) {
            if (i == b) {
              destDim[i] = srcDim[a];
            } else {
              g_assert (i < b && i < (NNS_TENSOR_RANK_LIMIT - 1));
              destDim[i] = srcDim[i + 1];
            }
          } else {
            /* a == b */
            destDim[i] = srcDim[i];
          }
        }
        ret = TRUE;
      } else {
        int i;
        int a = filter->data_dimchg.from;
        int b = filter->data_dimchg.to;

        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          if (i < a && i < b) {
            destDim[i] = srcDim[i];
          } else if (i > a && i > b) {
            destDim[i] = srcDim[i];
          } else if (a > b) {
            if (i == a) {
              destDim[i] = srcDim[b];
            } else {
              g_assert (i < a && i < (NNS_TENSOR_RANK_LIMIT - 1));
              destDim[i] = srcDim[i + 1];
            }
          } else if (a < b) {
            if (i == a) {
              destDim[i] = srcDim[b];
            } else {
              g_assert (i > 0 && i > a);
              destDim[i] = srcDim[i - 1];
            }
          } else {
            /* a == b */
            destDim[i] = srcDim[i];
          }
        }
        ret = TRUE;
      }
      break;
    default:
      return FALSE;
  }
  return ret;
}

/**
 * @brief configure srcpad cap from "proposed" cap. (required vmethod for BaseTransform)
 *
 * @param trans ("this" pointer)
 * @param direction (why do we need this?)
 * @param caps sinkpad cap
 * @param filtercap this element's cap (don't know specifically.)
 *
 * @todo Get to know what the heck is @filtercap and use it!
 */
static GstCaps *
gst_tensor_transform_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filtercap)
{
  /** @todo NYI: framerate configuration! */
  tensor_dim in, out;
  tensor_type type;
  gboolean ret;
  GstTensor_Transform *filter = GST_TENSOR_TRANSFORM_CAST (trans);
  GstCaps *retcap = NULL;
  gint fn = -1, fd = -1;

  g_assert (filter->loaded);

  if (direction == GST_PAD_SINK) {
    ret = gst_tensor_read_cap (caps, in, &type, &fn, &fd);
    if (ret == TRUE)
      ret = gst_tensor_dimension_conversion (filter, direction, in, out);
    if (ret == TRUE)
      retcap = gst_tensor_write_cap (out, type, fn, fd);
  } else {
    ret = gst_tensor_read_cap (caps, out, &type, &fn, &fd);
    if (ret == TRUE)
      ret = gst_tensor_dimension_conversion (filter, direction, out, in);
    if (ret == TRUE)
      retcap = gst_tensor_write_cap (in, type, fn, fd);
  }

  if (retcap == NULL) {
    /* Undetermined. Return the template pad */
    GstStaticCaps rawcap = GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT);
    GstCaps *staticcap = gst_static_caps_get (&rawcap);
    GstCaps *any = gst_caps_new_any ();
    retcap = gst_caps_intersect_full (staticcap, any, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (any);
  }

  if (filtercap) {
    GstCaps *retcap2 =
        gst_caps_intersect_full (retcap, filtercap, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (retcap);
    retcap = retcap2;
  }

  return retcap;
}

/**
 * @brief fixate caps. required vmethod of BaseTransform
 */
static GstCaps *
gst_tensor_transform_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  /** @todo NYI: framerate configuration! */
  GstCaps *othercaps_candidate =
      gst_tensor_transform_transform_caps (trans, direction, caps, NULL);
  GstCaps *filtered_candidate =
      gst_caps_intersect_full (othercaps_candidate, othercaps,
      GST_CAPS_INTERSECT_FIRST);

  gst_caps_unref (othercaps_candidate);

  return filtered_candidate;
}

/**
 * @brief set caps. required vmethod of BaseTransform
 */
static gboolean
gst_tensor_transform_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensor_Transform *filter = GST_TENSOR_TRANSFORM_CAST (trans);
  tensor_type type;
  gboolean ret;
  gint fd, fn;

  ret = gst_tensor_read_cap (incaps, filter->fromDim, &type, &fn, &fd);
  if (ret == FALSE)
    return FALSE;
  ret = gst_tensor_read_cap (outcaps, filter->toDim, &filter->type, &fd, &fn);
  if (ret == FALSE)
    return FALSE;
  if (type != filter->type)
    return FALSE;

  return TRUE;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. Note that this is always the same with the input. optional vmethod of BaseTransform
 */
static gboolean
gst_tensor_transform_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  *othersize = size;            /* size of input = size of output */
  return TRUE;

  /** @todo add verificastion procedure */
}
