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
 * @brief	GStreamer plugin to transform tensor dimension or type
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		This is NYI.
 *
 */

/**
 * SECTION:element-tensor_transform
 *
 * A filter that transforms tensor dimension or type.
 * The input and output is always in the format of other/tensor
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
#include <math.h>
#include "tensor_transform.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!filter->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) \
    debug_print (DBG, __VA_ARGS__)

#define silent_debug_caps(caps,msg) do { \
  if (DBG) { \
    if (caps) { \
      GstStructure *caps_s; \
      gchar *caps_s_string; \
      guint caps_size, caps_idx; \
      caps_size = gst_caps_get_size (caps);\
      for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
        caps_s = gst_caps_get_structure (caps, caps_idx); \
        caps_s_string = gst_structure_to_string (caps_s); \
        debug_print (TRUE, msg " = %s\n", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_transform_debug);
#define GST_CAT_DEFAULT gst_tensor_transform_debug

/**
 * @brief tensor_transform properties
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_MODE,
  PROP_OPTION,
};

static const gchar *gst_tensor_transform_mode_string[] = {
  [GTT_DIMCHG] = "dimchg",
  [GTT_TYPECAST] = "typecast",
  [GTT_ARITHMETIC] = "arithmetic",
  [GTT_TRANSPOSE] = "transpose",
  [GTT_STAND] = "stand",
  [GTT_END] = "error"
};

static const gchar *gst_tensor_transform_stand_string[] = {
  [STAND_DEFAULT] = "default",
  [STAND_END] = "error"
};

static const gchar *gst_tensor_transform_arithmetic_string[] = {
  [ARITH_ADD] = "add",
  [ARITH_MUL] = "mul",
  [ARITH_ADD_MUL] = "add-mul",
  [ARITH_MUL_ADD] = "mul-add",
  [ARITH_END] = "error"
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

#define gst_tensor_transform_parent_class parent_class
G_DEFINE_TYPE (GstTensorTransform, gst_tensor_transform,
    GST_TYPE_BASE_TRANSFORM);

/* GObject vmethod implementations */
static void gst_tensor_transform_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_transform_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_transform_finalize (GObject * object);

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

/**
 * @brief initialize the tensor_transform's class
 */
static void
gst_tensor_transform_class_init (GstTensorTransformClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_transform_set_property;
  gobject_class->get_property = gst_tensor_transform_get_property;
  gobject_class->finalize = gst_tensor_transform_finalize;

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
      "TensorTransform",
      "Converter/Filter/Tensor",
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
gst_tensor_transform_init (GstTensorTransform * filter)
{
  filter->silent = TRUE;
  filter->mode = GTT_END;
  filter->option = NULL;
  filter->loaded = FALSE;
  filter->type = _NNS_END;
}

/**
 * @brief Get the corresponding mode from the string value
 * @param[in] str The string value for the mode
 * @return corresponding mode for the string. ARITH_END for errors
 */
static tensor_transform_arith_mode
gst_tensor_transform_get_arith_mode (const gchar * str)
{
  int index;

  index = find_key_strv (gst_tensor_transform_arithmetic_string, str);

  return (index < 0) ? ARITH_END : index;
}

/**
 * @brief Get the corresponding mode from the string value
 * @param[in] str The string value for the mode
 * @return corresponding mode for the string. STAND_END for errors
 */
static tensor_transform_stand_mode
gst_tensor_transform_get_stand_mode (const gchar * str)
{
  int index;

  index = find_key_strv (gst_tensor_transform_stand_string, str);

  return (index < 0) ? STAND_END : index;
}

/**
 * @brief Get the corresponding mode from the string value
 * @param[in] str The string value for the mode
 * @return corresponding mode for the string. GTT_END for errors
 */
static tensor_transform_mode
gst_tensor_transform_get_mode (const gchar * str)
{
  int index;

  index = find_key_strv (gst_tensor_transform_mode_string, str);

  return (index < 0) ? GTT_END : index;
}

/**
 * @brief Setup internal data (data_* in GstTensorTransform)
 * @param[in/out] filter "this" pointer. mode & option MUST BE set already.
 */
static void
gst_tensor_transform_set_option_data (GstTensorTransform * filter)
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
      g_strfreev (strv);
      break;
    }
    case GTT_TYPECAST:
    {
      filter->data_typecast.to = get_tensor_type (filter->option);
      if (filter->data_typecast.to != _NNS_END)
        filter->loaded = TRUE;
      break;
    }
    case GTT_ARITHMETIC:
    {
      gchar **strv = g_strsplit (filter->option, ":", 2);

      if (strv[0] != NULL) {
        filter->data_arithmetic.mode =
            gst_tensor_transform_get_arith_mode (strv[0]);
        g_assert (filter->data_arithmetic.mode != ARITH_END);
      }

      if (strv[1] != NULL) {
        gchar **operands = g_strsplit (strv[1], ":", 2);
        gchar *not_consumed;
        int i;

        for (i = 0; i < ARITH_OPRND_NUM_LIMIT; ++i) {
          filter->data_arithmetic.value[i].type = ARITH_OPRND_TYPE_END;
          if ((operands[i] != NULL) && (strlen (operands[i]) != 0)) {
            if (strchr (operands[i], '.') || strchr (operands[i], 'e') ||
                strchr (operands[i], 'E')) {
              filter->data_arithmetic.value[i].type = ARITH_OPRND_TYPE_DOUBLE;
              filter->data_arithmetic.value[i].value_double =
                  g_ascii_strtod (operands[i], &not_consumed);
            } else {
              filter->data_arithmetic.value[i].type = ARITH_OPRND_TYPE_INT64;
              filter->data_arithmetic.value[i].value_int64 =
                  g_ascii_strtoll (operands[i], &not_consumed, 10);
            }

            if (strlen (not_consumed)) {
              g_printerr ("%s is not a valid integer or floating point value\n",
                  operands[i]);
              g_assert (0);
            }
          }
        }

        g_strfreev (operands);
      }

      filter->loaded = TRUE;
      g_strfreev (strv);
      break;
    }
    case GTT_TRANSPOSE:
    {
      int a, i;
      gchar **strv = g_strsplit (filter->option, ":", NNS_TENSOR_RANK_LIMIT);

      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        if (strv[i] != NULL)
          a = g_ascii_strtoull (strv[i], NULL, 10);
        else
          a = 0;
        filter->data_transpose.trans_order[i] = a;
      }

      filter->loaded = TRUE;
      g_strfreev (strv);
      break;
    }
    case GTT_STAND:
    {
      filter->data_stand.mode =
          gst_tensor_transform_get_stand_mode (filter->option);
      g_assert (filter->data_stand.mode != STAND_END);
      filter->loaded = TRUE;
      break;
    }
    default:
      g_printerr ("Cannot identify mode\n");
      g_assert (0);
      break;
  }
}

/**
 * @brief Set property (gst element vmethod)
 */
static void
gst_tensor_transform_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorTransform *filter = GST_TENSOR_TRANSFORM (object);

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
  GstTensorTransform *filter = GST_TENSOR_TRANSFORM (object);

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

/**
 * @brief Function to finalize instance (gst element vmethod)
 */
static void
gst_tensor_transform_finalize (GObject * object)
{
  GstTensorTransform *filter;

  filter = GST_TENSOR_TRANSFORM (object);

  if (filter->option) {
    g_free (filter->option);
    filter->option = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief subrouting for tensor-tranform, "dimchg" case.
 * @param[in/out] filter "this" pointer
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_dimchg (GstTensorTransform * filter,
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
 * Macro to run loop for various data types with simple cast
 */
#define castloop(itype,otype,num) do { \
    otype *ptr = (otype *) outptr; \
    itype *iptr = (itype *) inptr; \
    size_t i; \
    for (i = 0; i < num; i++) { \
      *(ptr + i) = (otype) *(iptr + i); \
    } \
  } while (0)

/**
 * Macro to run loop for various data types with simple cast
 * While castloop directly casts itype to otype, this macro indirectly casts
 * itype to otype using mtype as an intermediate
 */
#define castloop_via_intermediate(itype, mtype, otype, num) do { \
    otype *ptr = (otype *) outptr; \
    itype *iptr = (itype *) inptr; \
    size_t i; \
    for (i = 0; i < num; i++) { \
      mtype m = (mtype) *(iptr + i);\
      *(ptr + i) = (otype) m; \
    } \
  } while (0)

/**
 * Macro to run loop for various data types with a converter function
 */
#define convloop(itype,otype,num,convfunc) do { \
    otype *ptr = (otype *) outptr; \
    itype *iptr = (itype *) inptr; \
    size_t i; \
    for (i = 0; i < num; i++) { \
      *(ptr + i) = convfunc(iptr + i); \
    } \
  } while (0)

/**
 * Macro to unburden switch cases with castloop/convloop (per itype)
 * This is for cases otype is numeral.
 */
#define numotype_castloop_per_itype(otype,num) do { \
    switch (filter->type) { \
    case _NNS_INT8: castloop(int8_t, otype, num); break; \
    case _NNS_INT16: castloop(int16_t, otype, num); break; \
    case _NNS_INT32: castloop(int32_t, otype, num); break; \
    case _NNS_UINT8: castloop(uint8_t, otype, num); break; \
    case _NNS_UINT16: castloop(uint16_t, otype, num); break; \
    case _NNS_UINT32: castloop(uint32_t, otype, num); break; \
    case _NNS_FLOAT32: castloop(float, otype, num); break; \
    case _NNS_FLOAT64: castloop(double, otype, num); break; \
    case _NNS_INT64: castloop(int64_t, otype, num); break; \
    case _NNS_UINT64: castloop(uint64_t, otype, num); break; \
    default: g_assert(0); return GST_FLOW_ERROR; \
    } \
  } while (0)

#define numotype_castloop_via_intermediate_for_float_itype(mtype, otype, num) do { \
    switch (filter->type) { \
     case _NNS_FLOAT32:\
      castloop_via_intermediate(float, mtype, otype, num); \
      break; \
    case _NNS_FLOAT64: \
      castloop_via_intermediate(double, mtype, otype, num); \
      break; \
    default: g_assert(0); \
    } \
  } while (0)

/**
 * @brief subrouting for tensor-tranform, "typecast" case.
 * @param[in/out] filter "this" pointer
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_typecast (GstTensorTransform * filter,
    const uint8_t * inptr, uint8_t * outptr)
{
  uint32_t num = get_tensor_element_count (filter->fromDim);

  switch (filter->data_typecast.to) {
    case _NNS_INT8:
      numotype_castloop_per_itype (int8_t, num);
      break;
    case _NNS_INT16:
      numotype_castloop_per_itype (int16_t, num);
      break;
    case _NNS_INT32:
      numotype_castloop_per_itype (int32_t, num);
      break;
    case _NNS_UINT8:
      if ((filter->type == _NNS_FLOAT32) || (filter->type == _NNS_FLOAT64)) {
        numotype_castloop_via_intermediate_for_float_itype (int8_t, uint8_t,
            num);
      } else {
        numotype_castloop_per_itype (uint8_t, num);
      }
      break;
    case _NNS_UINT16:
      if ((filter->type == _NNS_FLOAT32) || (filter->type == _NNS_FLOAT64)) {
        numotype_castloop_via_intermediate_for_float_itype (int16_t, uint16_t,
            num);
      } else {
        numotype_castloop_per_itype (uint16_t, num);
      }
      break;
    case _NNS_UINT32:
      if ((filter->type == _NNS_FLOAT32) || (filter->type == _NNS_FLOAT64)) {
        numotype_castloop_via_intermediate_for_float_itype (int32_t, uint32_t,
            num);
      } else {
        numotype_castloop_per_itype (uint32_t, num);
      }
      break;
    case _NNS_FLOAT32:
      numotype_castloop_per_itype (float, num);
      break;
    case _NNS_FLOAT64:
      numotype_castloop_per_itype (double, num);
      break;
    case _NNS_INT64:
      numotype_castloop_per_itype (int64_t, num);
      break;
    case _NNS_UINT64:
      if ((filter->type == _NNS_FLOAT32) || (filter->type == _NNS_FLOAT64)) {
        numotype_castloop_via_intermediate_for_float_itype (int64_t, uint64_t,
            num);
      } else {
        numotype_castloop_per_itype (uint64_t, num);
      }
      break;
    default:
      g_assert (0);
      return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * Macro to run loop for various data types with simple arithmetic which has single operand
 */
#define arith(itype,num,op,a) do { \
    size_t i; \
    itype *in = (itype *) inptr; \
    itype *out = (itype *) outptr; \
    for (i=0;i<num;i++){ \
      *(out+i) = (*(in+i) op a); \
    } \
  }while(0);

/**
 * Macro to run loop for various data types with simple arithmetic which has dual operands
 */
#define arith2(itype,num,op1,a,op2,b) do { \
    size_t i; \
    itype *in = (itype *) inptr; \
    itype *out = (itype *) outptr; \
    for (i=0;i<num;i++){ \
      *(out+i) = (*(in+i) op1 a) op2 b; \
    } \
  }while(0);

/**
 * Macro to handle the case of single operand
 */
#define arithmode_single_oprnd_case(itype,num,mode,op,value) do { \
  itype a;\
  switch (value[0].type) {\
  case ARITH_OPRND_TYPE_INT64 : a = (itype) value[0].value_int64; break; \
  case ARITH_OPRND_TYPE_DOUBLE : a = (itype) value[0].value_double; break;\
  default: \
  g_printerr ("The operand required by \'%s\' is not properly provided.\n", \
      gst_tensor_transform_arithmetic_string[filter->data_arithmetic.mode]);\
  g_assert(0); \
  }; \
  arith(itype, num, op, a); break; \
} while (0);

/**
 * Macro to handle the case of dual operands
 */
#define arithmode_dual_oprnd_case(itype,num,mode,op1,op2,value) \
do {\
  itype a;\
  itype b; \
  switch (value[0].type) {\
  case ARITH_OPRND_TYPE_INT64 : a = (itype) value[0].value_int64; break; \
  case ARITH_OPRND_TYPE_DOUBLE : a = (itype) value[0].value_double; break;\
  default: \
  g_printerr ("The operands required by \'%s\' are not properly provided.\n", \
      gst_tensor_transform_arithmetic_string[filter->data_arithmetic.mode]);\
  g_assert(0); \
  }; \
  switch (value[1].type) {\
  case ARITH_OPRND_TYPE_INT64 : b = (itype) value[1].value_int64; break; \
  case ARITH_OPRND_TYPE_DOUBLE : b = (itype) value[1].value_double; break;\
  default: \
  g_printerr ("The operands required by \'%s\' are not properly provided.\n", \
      gst_tensor_transform_arithmetic_string[filter->data_arithmetic.mode]);\
  g_assert(0); \
  }; \
  arith2(itype, num, op1, a, op2, b); break; \
} while (0);

/**
 * Macro to run loop for various data types with simple arithmetic
 */
#define arithloopcase(typecase,itype,num,mode,value) \
  case typecase: \
  { \
    switch (mode) { \
    case ARITH_ADD: {\
      arithmode_single_oprnd_case (itype, num, mode, +, value); \
      break; \
    }; \
    case ARITH_MUL: { \
      arithmode_single_oprnd_case (itype, num, mode, *, value); \
      break; \
    };\
    case ARITH_ADD_MUL: {\
      arithmode_dual_oprnd_case (itype, num, mode, +, *, value); \
      break; \
    }; \
    case ARITH_MUL_ADD: {\
      arithmode_dual_oprnd_case (itype, num, mode, *, +, value); \
      break; \
    }; \
    default: g_assert(0); return GST_FLOW_ERROR; \
    } \
    break; \
  }

/**
 * @brief subrouting for tensor-tranform, "arithmetic" case.
 * @param[in/out] filter "this" pointer
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_arithmetic (GstTensorTransform * filter,
    const uint8_t * inptr, uint8_t * outptr)
{
  uint32_t num = get_tensor_element_count (filter->fromDim);
  tensor_transform_arith_mode mode = filter->data_arithmetic.mode;
  tensor_transform_arithmetic_operand *value = filter->data_arithmetic.value;

  switch (filter->type) {
      arithloopcase (_NNS_INT8, int8_t, num, mode, value);
      arithloopcase (_NNS_INT16, int16_t, num, mode, value);
      arithloopcase (_NNS_INT32, int32_t, num, mode, value);
      arithloopcase (_NNS_UINT8, uint8_t, num, mode, value);
      arithloopcase (_NNS_UINT16, uint16_t, num, mode, value);
      arithloopcase (_NNS_UINT32, uint32_t, num, mode, value);
      arithloopcase (_NNS_FLOAT32, float, num, mode, value);
      arithloopcase (_NNS_FLOAT64, double, num, mode, value);
      arithloopcase (_NNS_INT64, int64_t, num, mode, value);
      arithloopcase (_NNS_UINT64, uint64_t, num, mode, value);
    default:
      g_assert (0);
      return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * Macro to run loop for various data types with transpose
 */
#define transposeloop(cl,ck,cj,ci,sl,sk,sj,si,typesize) do { \
    size_t i, j, k, l;                                  \
    int inidx = 0, outidx=0;                            \
    for(cl=0;cl<sl;cl++)                      \
      for(ci=0;ci<si;ci++)                    \
	for(cj=0;cj<sj;cj++)                  \
	  for(ck=0;ck<sk;ck++){               \
	    outidx=si*sj*sk*cl + sj*sk*ci + sk*cj+ck; \
	    inidx = SK*SJ*SI*l + SJ*SI*k + SI*j + i; \
	    const uint8_t *_in = inptr+inidx*typesize; \
	    uint8_t *_out = outptr + outidx *typesize; \
	    memcpy(_out, _in, typesize); \
	  }                                                      \
  } while(0);

/**
 * @brief subrouting for tensor-tranform, "transpose" case.
 * @param[in/out] filter "this" pointer
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_transpose (GstTensorTransform * filter,
    const uint8_t * inptr, uint8_t * outptr)
{
  int i, from, to;
  gboolean checkdim = FALSE;
  uint32_t *fromDim = filter->fromDim;
  size_t type_size = tensor_element_size[filter->type];
  size_t indexI, indexJ, SL, SI, SJ, SK;
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    from = i;
    to = filter->data_transpose.trans_order[i];
    if (from != to) {
      checkdim = TRUE;
      break;
    }
  }

  if (!checkdim) {
    memcpy (outptr, inptr,
        get_tensor_element_count (filter->fromDim) * type_size);
    g_printerr
        ("Calling tensor_transform with high memcpy overhead WITHOUT any effects!");
    return GST_FLOW_OK;
  }

  indexI = filter->data_transpose.trans_order[0];
  indexJ = filter->data_transpose.trans_order[1];
  SL = fromDim[3], SI = fromDim[0], SJ = fromDim[1], SK = fromDim[2];

  g_assert (filter->data_transpose.trans_order[3] == 3);

  switch (indexI) {
    case 0:
      if (indexJ == 1) {
        transposeloop (l, i, j, k, SL, SI, SJ, SK, type_size);
      } else {
        transposeloop (l, i, k, j, SL, SI, SK, SJ, type_size);
      }
      break;
    case 1:
      if (indexJ == 0) {
        transposeloop (l, j, i, k, SL, SJ, SI, SK, type_size);
      } else {
        transposeloop (l, j, k, i, SL, SJ, SK, SI, type_size);
      }
      break;
    case 2:
      if (indexJ == 0) {
        transposeloop (l, k, i, j, SL, SK, SI, SJ, type_size);
      } else {
        transposeloop (l, k, j, i, SL, SK, SJ, SI, type_size);
      }
      break;
  }

  return GST_FLOW_OK;
}

/**
 * @brief subrouting for tensor-tranform, "stand" case.
 *        : pixel = abs((pixel - average(tensor))/(std(tensor) + val))
 * @param[in/out] filter "this" pointer
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_stand (GstTensorTransform * filter,
    const uint8_t * inptr, uint8_t * outptr)
{
  int i;
  size_t Size;
  uint32_t *fromDim = filter->fromDim;
  double average, stand;

  float *in = (float *) inptr;
  float *out = (float *) outptr;
  Size = fromDim[3] * fromDim[2] * fromDim[1] * fromDim[0];

  switch (filter->data_stand.mode) {
    case STAND_DEFAULT:
    {
      average = 0.0;

      for (i = 0; i < Size; i++) {
        average = (in[i] - average) / (i + 1) + average;
      }

      stand = 0.0;

      for (i = 0; i < Size; i++) {
        stand += pow (in[i] - average, 2) / (Size - 1);
      }

      stand = sqrt (stand);
      for (i = 0; i < Size; i++) {
        out[i] = fabs ((in[i] - average) / (stand + 1e-10));
      }

      break;
    }
    default:
      g_printerr ("Cannot identify mode\n");
      g_assert (0);
      return GST_FLOW_ERROR;
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
  GstTensorTransform *filter = GST_TENSOR_TRANSFORM_CAST (trans);

  uint8_t *inptr, *outptr;
  GstMapInfo inInfo, outInfo;

  g_assert (gst_buffer_map (inbuf, &inInfo, GST_MAP_READ));
  g_assert (gst_buffer_map (outbuf, &outInfo, GST_MAP_WRITE));

  inptr = inInfo.data;
  outptr = outInfo.data;

  switch (filter->mode) {
    case GTT_DIMCHG:
      res = gst_tensor_transform_dimchg (filter, inptr, outptr);
      break;
    case GTT_TYPECAST:
      res = gst_tensor_transform_typecast (filter, inptr, outptr);
      break;
    case GTT_ARITHMETIC:
      res = gst_tensor_transform_arithmetic (filter, inptr, outptr);
      break;
    case GTT_TRANSPOSE:
      res = gst_tensor_transform_transpose (filter, inptr, outptr);
      break;
    case GTT_STAND:
      res = gst_tensor_transform_stand (filter, inptr, outptr);
      break;
    default:
      res = GST_FLOW_NOT_SUPPORTED;
      break;
  }

  gst_buffer_unmap (inbuf, &inInfo);
  gst_buffer_unmap (outbuf, &outInfo);

  return res;
}

/**
 * @brief Read cap, parse tensor configuration (dim/type) from the cap.
 * @param[in] caps The input caps to be read
 * @param[out] config configured tensor info
 * @return TRUE if successful (both dim/type read). FALSE if not.
 */
static gboolean
gst_tensor_transform_read_caps (const GstCaps * caps, GstTensorConfig * config)
{
  GstStructure *structure;

  g_return_val_if_fail (config != NULL, FALSE);

  structure = gst_caps_get_structure (caps, 0);

  if (!gst_structure_has_name (structure, "other/tensor")) {
    err_print ("caps is not tensor %s\n", gst_structure_get_name (structure));
    return FALSE;
  }

  gst_tensor_config_from_structure (config, structure);

  return gst_tensor_info_validate (&config->info);
}

/**
 * @brief Dimension conversion calculation
 * @param[in] filter "this" pointer
 * @param[in] direction GST_PAD_SINK if input->output conv
 * @param[in] in_info tensor info structure of source tensor (input if direction is SINK)
 * @param[out] out_info tensor info structure of destination tensor (output if direction is SINK)
 * @return TRUE if success
 */
static gboolean
gst_tensor_transform_convert_dimension (GstTensorTransform * filter,
    GstPadDirection direction, const GstTensorInfo * in_info,
    GstTensorInfo * out_info)
{
  switch (filter->mode) {
    case GTT_DIMCHG:
      out_info->type = in_info->type;

      if (direction == GST_PAD_SINK) {
        int i;
        int a = filter->data_dimchg.from;
        int b = filter->data_dimchg.to;

        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          if (i < a && i < b) {
            out_info->dimension[i] = in_info->dimension[i];
          } else if (i > a && i > b) {
            out_info->dimension[i] = in_info->dimension[i];
          } else if (a > b) {
            if (i == b) {
              out_info->dimension[i] = in_info->dimension[a];
            } else {
              g_assert (i > 0 && i > b);
              out_info->dimension[i] = in_info->dimension[i - 1];
            }
          } else if (a < b) {
            if (i == b) {
              out_info->dimension[i] = in_info->dimension[a];
            } else {
              g_assert (i < b && i < (NNS_TENSOR_RANK_LIMIT - 1));
              out_info->dimension[i] = in_info->dimension[i + 1];
            }
          } else {
            /* a == b */
            out_info->dimension[i] = in_info->dimension[i];
          }
        }
      } else {
        int i;
        int a = filter->data_dimchg.from;
        int b = filter->data_dimchg.to;

        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          if (i < a && i < b) {
            out_info->dimension[i] = in_info->dimension[i];
          } else if (i > a && i > b) {
            out_info->dimension[i] = in_info->dimension[i];
          } else if (a > b) {
            if (i == a) {
              out_info->dimension[i] = in_info->dimension[b];
            } else {
              g_assert (i < a && i < (NNS_TENSOR_RANK_LIMIT - 1));
              out_info->dimension[i] = in_info->dimension[i + 1];
            }
          } else if (a < b) {
            if (i == a) {
              out_info->dimension[i] = in_info->dimension[b];
            } else {
              g_assert (i > 0 && i > a);
              out_info->dimension[i] = in_info->dimension[i - 1];
            }
          } else {
            /* a == b */
            out_info->dimension[i] = in_info->dimension[i];
          }
        }
      }
      break;
    case GTT_TYPECAST:
    {
        /** For both directions, dimension does not change */
      int i;
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        out_info->dimension[i] = in_info->dimension[i];
      }
      if (direction == GST_PAD_SINK) {
          /** src = SINKPAD / dest = SRCPAD */
        out_info->type = filter->data_typecast.to;
      } else {
          /** src = SRCPAD / dest = SINKPAD */
        out_info->type = filter->type;   /** @todo this may cause problems with Cap-Transform */
      }
      break;
    }
    case GTT_ARITHMETIC:
    {
      int i;
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        out_info->dimension[i] = in_info->dimension[i];
      }
      out_info->type = in_info->type;
      break;
    }
    case GTT_TRANSPOSE:
    {
      out_info->type = in_info->type;
      int i;
      if (direction == GST_PAD_SINK) {
        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          out_info->dimension[i] =
              in_info->dimension[filter->data_transpose.trans_order[i]];
        }
      } else {
        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          g_assert (filter->data_transpose.trans_order[i] <
              NNS_TENSOR_RANK_LIMIT);
          out_info->dimension[filter->data_transpose.trans_order[i]] =
              in_info->dimension[i];
        }
      }
      break;
    }
    case GTT_STAND:
    {
      int i;
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        out_info->dimension[i] = in_info->dimension[i];
      }
      out_info->type = in_info->type;
      break;
    }
    default:
      return FALSE;
  }

  return TRUE;
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
  GstTensorTransform *filter;
  GstTensorConfig in_config;
  GstTensorConfig out_config;
  GstCaps *result = NULL;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);
  g_assert (filter->loaded);

  silent_debug ("Calling TransformCaps, direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filtercap, "filter");

  gst_tensor_config_init (&in_config);
  gst_tensor_config_init (&out_config);

  if (direction == GST_PAD_SINK) {
    if (gst_tensor_transform_read_caps (caps, &in_config)) {
      gst_tensor_transform_convert_dimension (filter, direction,
          &in_config.info, &out_config.info);
    }

    /**
     * supposed same framerate from input configuration
     */
    out_config.rate_n = in_config.rate_n;
    out_config.rate_d = in_config.rate_d;

    result = gst_tensor_caps_from_config (&out_config);
  } else {
    if (gst_tensor_transform_read_caps (caps, &out_config)) {
      gst_tensor_transform_convert_dimension (filter, direction,
          &out_config.info, &in_config.info);
    }

    result = gst_tensor_caps_from_config (&in_config);
  }

  if (filtercap) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (result, filtercap, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (result, "to");
  return result;
}

/**
 * @brief fixate caps. required vmethod of BaseTransform
 */
static GstCaps *
gst_tensor_transform_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstTensorTransform *filter;
  GstCaps *result;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug ("Calling FixateCaps, direction = %d\n", direction);
  silent_debug_caps (caps, "caps");
  silent_debug_caps (othercaps, "othercaps");

  result = gst_tensor_transform_transform_caps (trans, direction, caps, NULL);

  if (othercaps) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (result, othercaps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (othercaps);
    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (result, "result");
  return result;
}

/**
 * @brief set caps. required vmethod of BaseTransform
 */
static gboolean
gst_tensor_transform_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensorTransform *filter;
  GstTensorConfig in_config;
  GstTensorConfig out_config;
  guint i;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug ("Calling SetCaps\n");
  silent_debug_caps (incaps, "incaps");
  silent_debug_caps (outcaps, "outcaps");

  if (!gst_tensor_transform_read_caps (incaps, &in_config) ||
      !gst_tensor_config_validate (&in_config)) {
    silent_debug ("Cannot read cap of incaps\n");
    goto error;
  }

  if (!gst_tensor_transform_read_caps (outcaps, &out_config) ||
      !gst_tensor_config_validate (&out_config)) {
    silent_debug ("Cannot read cap of outcaps\n");
    goto error;
  }

  /** check framerate */
  if (in_config.rate_n != out_config.rate_n
      || in_config.rate_d != out_config.rate_d) {
    silent_debug ("Framerate is not matched\n");
    goto error;
  }

  /**
   * Update in/out tensor info (dimension, type)
   */
  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    filter->fromDim[i] = in_config.info.dimension[i];
    filter->toDim[i] = out_config.info.dimension[i];
  }

  if (filter->type == _NNS_END)
    filter->type = in_config.info.type;

  switch (filter->mode) {
    case GTT_TRANSPOSE:
    case GTT_ARITHMETIC:
    case GTT_STAND:
    case GTT_DIMCHG:
      if (in_config.info.type != out_config.info.type
          || filter->type != in_config.info.type) {
        silent_debug ("Filter Type Not Matched\n");
        goto error;
      }
      break;
    case GTT_TYPECAST:
      if (filter->type != in_config.info.type
          || filter->data_typecast.to != out_config.info.type) {
        silent_debug ("Filter Type Not Matched\n Input %d/%d | Output %d/%d",
            filter->type, in_config.info.type, filter->data_typecast.to,
            out_config.info.type);
        goto error;
      }
      break;
    default:
      break;
  }

  return TRUE;
error:
  silent_debug ("Set Caps Failed!\n");
  return FALSE;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. Note that this is always the same with the input. optional vmethod of BaseTransform
 */
static gboolean
gst_tensor_transform_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensorTransform *filter;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  switch (filter->mode) {
    case GTT_TRANSPOSE:
    case GTT_ARITHMETIC:
    case GTT_STAND:
    case GTT_DIMCHG:
      *othersize = size;        /* size of input = size of output if dimchg */
      break;
    case GTT_TYPECAST:
    {
      size_t srcunitsize = tensor_element_size[filter->type];
      size_t dstunitsize = tensor_element_size[filter->data_typecast.to];
      if (size % srcunitsize > 0)
        return FALSE;
      *othersize = size / srcunitsize * dstunitsize;
      break;
    }
    default:
      return FALSE;
  }
  return TRUE;

  /** @todo add verificastion procedure */
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
NNSTREAMER_PLUGIN_INIT (tensor_transform)
{
  /**
   * debug category for fltering log messages
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_transform_debug, "tensor_transform",
      0, "tensor_transform element");

  return gst_element_register (plugin, "tensor_transform",
      GST_RANK_NONE, GST_TYPE_TENSOR_TRANSFORM);
}

#ifndef SINGLE_BINARY
/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

/**
 * gstreamer looks for this structure to register tensor_transforms
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_transform,
    "GStreamer plugin to transform tensor dimension or type",
    gst_tensor_transform_plugin_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer");
#endif
