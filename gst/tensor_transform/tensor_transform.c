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

#ifdef HAVE_ORC
#include "transform-orc.h"
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!filter->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (filter, __VA_ARGS__); \
    } \
  } while (0)

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
        GST_DEBUG_OBJECT (filter, msg " = %s\n", caps_s_string); \
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
  PROP_ACCELERATION
};

/**
 * @brief Flag to set orc acceleration.
 */
#ifdef HAVE_ORC
#define DEFAULT_ACCELERATION TRUE
#else
#define DEFAULT_ACCELERATION FALSE
#endif

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

static const gchar *gst_tensor_transform_operator_string[] = {
  [GTT_OP_TYPECAST] = "typecast",
  [GTT_OP_ADD] = "add",
  [GTT_OP_MUL] = "mul",
  [GTT_OP_DIV] = "div",
  [GTT_OP_UNKNOWN] = "unknown"
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
  g_object_class_install_property (gobject_class, PROP_ACCELERATION,
      g_param_spec_boolean ("acceleration", "Acceleration", "Orc acceleration",
          DEFAULT_ACCELERATION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  filter->operators = NULL;
  filter->acceleration = DEFAULT_ACCELERATION;
#ifdef HAVE_ORC
  filter->orc_supported = FALSE;
#endif

  gst_tensor_config_init (&filter->in_config);
  gst_tensor_config_init (&filter->out_config);
}

/**
 * @brief Get the corresponding operator from the string value
 * @param[in] str The string value for the operator
 * @return corresponding operator for the string (GTT_OP_UNKNOWN for errors)
 */
static tensor_transform_operator
gst_tensor_transform_get_operator (const gchar * str)
{
  int index;

  index = find_key_strv (gst_tensor_transform_operator_string, str);

  return (index < 0) ? GTT_OP_UNKNOWN : index;
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

#ifdef HAVE_ORC
/* define macros for orc */
#define orc_supported(filter) (filter->acceleration && filter->orc_supported)

#define orc_func_conv(intype,outtype) nns_orc_conv_ ## intype ## _to_ ## outtype
#define orc_func_add(intype) nns_orc_add_c_ ## intype
#define orc_func_mul(intype) nns_orc_mul_c_ ## intype
#define orc_func_div(intype) nns_orc_div_c_ ## intype

#define orc_typecast_to(i,o,n,intype,otype) do { \
    switch (otype) { \
      case _NNS_INT32: orc_func_conv (intype, s32) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_UINT32: orc_func_conv (intype, u32) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_INT16: orc_func_conv (intype, s16) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_UINT16: orc_func_conv (intype, u16) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_INT8: orc_func_conv (intype, s8) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_UINT8: orc_func_conv (intype, u8) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_FLOAT64: orc_func_conv (intype, f64) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_FLOAT32: orc_func_conv (intype, f32) ((gpointer) o, (gpointer) i, n); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported type %d", otype); g_assert (0); break; \
    } \
  } while (0)

#define orc_typecast(i,o,n,itype,otype) do { \
    switch (itype) { \
      case _NNS_INT32: orc_typecast_to (i, o, n, s32, otype); break; \
      case _NNS_UINT32: orc_typecast_to (i, o, n, u32, otype); break; \
      case _NNS_INT16: orc_typecast_to (i, o, n, s16, otype); break; \
      case _NNS_UINT16: orc_typecast_to (i, o, n, u16, otype); break; \
      case _NNS_INT8: orc_typecast_to (i, o, n, s8, otype); break; \
      case _NNS_UINT8: orc_typecast_to (i, o, n, u8, otype); break; \
      case _NNS_FLOAT64: orc_typecast_to (i, o, n, f64, otype); break; \
      case _NNS_FLOAT32: orc_typecast_to (i, o, n, f32, otype); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported type %d", itype); g_assert (0); break; \
    } \
  } while (0)

#define orc_operator_func(i,o,n,v,opfunc) do { \
    switch ((v)->type) { \
      case _NNS_INT32: opfunc (s32) ((gpointer) o, (gpointer) i, (v)->data._int32_t, n); break; \
      case _NNS_UINT32: opfunc (u32) ((gpointer) o, (gpointer) i, (v)->data._uint32_t, n); break; \
      case _NNS_INT16: opfunc (s16) ((gpointer) o, (gpointer) i, (v)->data._int16_t, n); break; \
      case _NNS_UINT16: opfunc (u16) ((gpointer) o, (gpointer) i, (v)->data._uint16_t, n); break; \
      case _NNS_INT8: opfunc (s8) ((gpointer) o, (gpointer) i, (v)->data._int8_t, n); break; \
      case _NNS_UINT8: opfunc (u8) ((gpointer) o, (gpointer) i, (v)->data._uint8_t, n); break; \
      case _NNS_FLOAT64: opfunc (f64) ((gpointer) o, (gpointer) i, (v)->data._double, n); break; \
      case _NNS_FLOAT32: opfunc (f32) ((gpointer) o, (gpointer) i, (v)->data._float, n); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported type %d", (v)->type); g_assert (0); break; \
    } \
  } while (0)

#define orc_operator_div_loop(i,o,n,val,typename) do { \
    gsize idx; \
    typename *data_in = (typename *) (i); \
    typename *data_out = (typename *) (o); \
    for (idx = 0; idx < (n); ++idx) { \
      data_out[idx] = data_in[idx] / (val); \
    } \
  } while (0)

#define orc_operator(i,o,n,v,op) do { \
    switch (op) { \
      case GTT_OP_ADD: orc_operator_func (i, o, n, v, orc_func_add); break; \
      case GTT_OP_MUL: orc_operator_func (i, o, n, v, orc_func_mul); break; \
      case GTT_OP_DIV: \
        switch ((v)->type) { \
          case _NNS_INT32: orc_operator_div_loop (i, o, n, (v)->data._int32_t, int32_t); break; \
          case _NNS_UINT32: orc_operator_div_loop (i, o, n, (v)->data._uint32_t, uint32_t); break; \
          case _NNS_INT16: orc_operator_div_loop (i, o, n, (v)->data._int16_t, int16_t); break; \
          case _NNS_UINT16: orc_operator_div_loop (i, o, n, (v)->data._uint16_t, uint16_t); break; \
          case _NNS_INT8: orc_operator_div_loop (i, o, n, (v)->data._int8_t, int8_t); break; \
          case _NNS_UINT8: orc_operator_div_loop (i, o, n, (v)->data._uint8_t, uint8_t); break; \
          case _NNS_FLOAT64: orc_func_div (f64) ((gpointer) o, (gpointer) i, (v)->data._double, n); break; \
          case _NNS_FLOAT32: orc_func_div (f32) ((gpointer) o, (gpointer) i, (v)->data._float, n); break; \
          default: GST_ERROR_OBJECT (filter, "Unsupported type %d", (v)->type); g_assert (0); break; \
        } \
        break; \
      default: GST_ERROR_OBJECT (filter, "Unknown operator %d", op); break; \
    } \
  } while (0)
#endif /* HAVE_ORC */

/**
 * @brief Macro to set operand
 */
#define set_operand_value(v,d,vtype) do { \
    (v)->data._##vtype = *((vtype *) d); \
  } while (0)

/**
 * @brief Set tensor element value with given type
 * @param filter "this" pointer
 * @param value struct for operand of arith mode
 * @param type tensor type
 * @param data pointer of tensor element value
 * @return TRUE if no error
 */
static gboolean
gst_tensor_transform_set_value (GstTensorTransform * filter,
    tensor_transform_operand_s * value, tensor_type type, gpointer data)
{
  g_return_val_if_fail (value != NULL, FALSE);
  g_return_val_if_fail (data != NULL, FALSE);

  /* init tensor value */
  memset (value, 0, sizeof (tensor_transform_operand_s));
  value->type = _NNS_END;

  switch (type) {
    case _NNS_INT32:
      set_operand_value (value, data, int32_t);
      break;
    case _NNS_UINT32:
      set_operand_value (value, data, uint32_t);
      break;
    case _NNS_INT16:
      set_operand_value (value, data, int16_t);
      break;
    case _NNS_UINT16:
      set_operand_value (value, data, uint16_t);
      break;
    case _NNS_INT8:
      set_operand_value (value, data, int8_t);
      break;
    case _NNS_UINT8:
      set_operand_value (value, data, uint8_t);
      break;
    case _NNS_FLOAT64:
      set_operand_value (value, data, double);
      break;
    case _NNS_FLOAT32:
      set_operand_value (value, data, float);
      break;
    case _NNS_INT64:
      set_operand_value (value, data, int64_t);
      break;
    case _NNS_UINT64:
      set_operand_value (value, data, uint64_t);
      break;
    default:
      GST_ERROR_OBJECT (filter, "Unknown tensor type %d", type);
      return FALSE;
  }

  value->type = type;
  return TRUE;
}

/**
 * @brief Macro to get operand
 */
#define get_operand_value(v,d,vtype) do { \
    *((vtype *) d) = (v)->data._##vtype; \
  } while (0)

/**
 * @brief Get tensor element value with given type
 * @param filter "this" pointer
 * @param value struct for operand of arith mode
 * @param data pointer of tensor element value
 * @return TRUE if no error
 */
static gboolean
gst_tensor_transform_get_value (GstTensorTransform * filter,
    tensor_transform_operand_s * value, gpointer data)
{
  g_return_val_if_fail (value != NULL, FALSE);
  g_return_val_if_fail (data != NULL, FALSE);

  switch (value->type) {
    case _NNS_INT32:
      get_operand_value (value, data, int32_t);
      break;
    case _NNS_UINT32:
      get_operand_value (value, data, uint32_t);
      break;
    case _NNS_INT16:
      get_operand_value (value, data, int16_t);
      break;
    case _NNS_UINT16:
      get_operand_value (value, data, uint16_t);
      break;
    case _NNS_INT8:
      get_operand_value (value, data, int8_t);
      break;
    case _NNS_UINT8:
      get_operand_value (value, data, uint8_t);
      break;
    case _NNS_FLOAT64:
      get_operand_value (value, data, double);
      break;
    case _NNS_FLOAT32:
      get_operand_value (value, data, float);
      break;
    case _NNS_INT64:
      get_operand_value (value, data, int64_t);
      break;
    case _NNS_UINT64:
      get_operand_value (value, data, uint64_t);
      break;
    default:
      GST_ERROR_OBJECT (filter, "Unknown tensor type %d", value->type);
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Macro for operator
 */
#define handle_operator(d,v,oper,vtype) do { \
    switch (oper) { \
      case GTT_OP_ADD: \
        (d)->data._##vtype += (v)->data._##vtype; \
        break; \
      case GTT_OP_MUL: \
        (d)->data._##vtype *= (v)->data._##vtype; \
        break; \
      case GTT_OP_DIV: \
        if ((v)->data._##vtype == 0) { \
          GST_ERROR_OBJECT (filter, "Invalid state, denominator is 0."); \
          return FALSE; \
        } \
        (d)->data._##vtype /= (v)->data._##vtype; \
        break; \
      default: \
        GST_ERROR_OBJECT (filter, "Unknown operator %d", oper); \
        return FALSE; \
    } \
  } while (0)

/**
 * @brief Handle operators for tensor value
 * @param filter "this" pointer
 * @param desc struct for tensor value
 * @param val struct for tensor value
 * @param op operator for given tensor value
 * @return TRUE if no error
 */
static gboolean
gst_tensor_transform_do_operator (GstTensorTransform * filter,
    tensor_transform_operand_s * desc, const tensor_transform_operand_s * val,
    tensor_transform_operator op)
{
  g_return_val_if_fail (desc != NULL, FALSE);
  g_return_val_if_fail (val != NULL, FALSE);
  g_return_val_if_fail (desc->type == val->type, FALSE);

  switch (desc->type) {
    case _NNS_INT32:
      handle_operator (desc, val, op, int32_t);
      break;
    case _NNS_UINT32:
      handle_operator (desc, val, op, uint32_t);
      break;
    case _NNS_INT16:
      handle_operator (desc, val, op, int16_t);
      break;
    case _NNS_UINT16:
      handle_operator (desc, val, op, uint16_t);
      break;
    case _NNS_INT8:
      handle_operator (desc, val, op, int8_t);
      break;
    case _NNS_UINT8:
      handle_operator (desc, val, op, uint8_t);
      break;
    case _NNS_FLOAT64:
      handle_operator (desc, val, op, double);
      break;
    case _NNS_FLOAT32:
      handle_operator (desc, val, op, float);
      break;
    case _NNS_INT64:
      handle_operator (desc, val, op, int64_t);
      break;
    case _NNS_UINT64:
      handle_operator (desc, val, op, uint64_t);
      break;
    default:
      GST_ERROR_OBJECT (filter, "Unknown tensor type %d", desc->type);
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Macro for typecast
 */
#define typecast_value_to(v,itype,otype) do { \
    itype in_val = (v)->data._##itype; \
    otype out_val = (otype) in_val; \
    (v)->data._##otype = out_val; \
  } while (0)

#define typecast_value(v,otype) do { \
    switch ((v)->type) { \
      case _NNS_INT32: typecast_value_to (v, int32_t, otype); break; \
      case _NNS_UINT32: typecast_value_to (v, uint32_t, otype); break; \
      case _NNS_INT16: typecast_value_to (v, int16_t, otype); break; \
      case _NNS_UINT16:  typecast_value_to (v, uint16_t, otype); break; \
      case _NNS_INT8: typecast_value_to (v, int8_t, otype); break; \
      case _NNS_UINT8: typecast_value_to (v, uint8_t, otype); break; \
      case _NNS_FLOAT64: typecast_value_to (v, double, otype); break; \
      case _NNS_FLOAT32: typecast_value_to (v, float, otype); break; \
      case _NNS_INT64: typecast_value_to (v, int64_t, otype); break; \
      case _NNS_UINT64: typecast_value_to (v, uint64_t, otype); break; \
      default: g_assert (0); break; \
    } \
  } while (0)

/**
 * @brief Typecast tensor element value
 * @param filter "this" pointer
 * @param value struct for operand of arith mode
 * @param type tensor type to be transformed
 * @return TRUE if no error
 */
static gboolean
gst_tensor_transform_typecast_value (GstTensorTransform * filter,
    tensor_transform_operand_s * value, tensor_type type)
{
  gboolean is_float;

  g_return_val_if_fail (value != NULL, FALSE);
  g_return_val_if_fail (type != _NNS_END, FALSE);

  /* do nothing when transform to same type */
  if (value->type != type) {
    is_float = (value->type == _NNS_FLOAT32 || value->type == _NNS_FLOAT64);

    switch (type) {
      case _NNS_INT32:
        typecast_value (value, int32_t);
        break;
      case _NNS_UINT32:
        if (is_float) {
          /* int32 -> uint32 */
          typecast_value (value, int32_t);
          value->type = _NNS_INT32;
        }
        typecast_value (value, uint32_t);
        break;
      case _NNS_INT16:
        typecast_value (value, int16_t);
        break;
      case _NNS_UINT16:
        if (is_float) {
          /* int16 -> uint16 */
          typecast_value (value, int16_t);
          value->type = _NNS_INT16;
        }
        typecast_value (value, uint16_t);
        break;
      case _NNS_INT8:
        typecast_value (value, int8_t);
        break;
      case _NNS_UINT8:
        if (is_float) {
          /* int8 -> uint8 */
          typecast_value (value, int8_t);
          value->type = _NNS_INT8;
        }
        typecast_value (value, uint8_t);
        break;
      case _NNS_FLOAT64:
        typecast_value (value, double);
        break;
      case _NNS_FLOAT32:
        typecast_value (value, float);
        break;
      case _NNS_INT64:
        typecast_value (value, int64_t);
        break;
      case _NNS_UINT64:
        if (is_float) {
          /* int64 -> uint64 */
          typecast_value (value, int64_t);
          value->type = _NNS_INT64;
        }
        typecast_value (value, uint64_t);
        break;
      default:
        GST_ERROR_OBJECT (filter, "Unknown tensor type %d", type);
        return FALSE;
    }

    value->type = type;
  }

  return TRUE;
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
      gchar **str_operators;
      gchar **str_op;
      tensor_transform_operator_s *op_s;
      guint i, num_operators, num_op;

      filter->data_arithmetic.out_type = _NNS_END;

      if (filter->operators) {
        GST_WARNING_OBJECT (filter,
            "There exists pre-defined operators (total %d), now reset these.",
            g_slist_length (filter->operators));

        g_slist_free_full (filter->operators, g_free);
        filter->operators = NULL;
      }

      str_operators = g_strsplit (filter->option, ",", -1);
      num_operators = g_strv_length (str_operators);

      for (i = 0; i < num_operators; ++i) {
        str_op = g_strsplit (str_operators[i], ":", -1);
        num_op = g_strv_length (str_op);

        if (str_op[0]) {
          op_s = g_new0 (tensor_transform_operator_s, 1);
          g_assert (op_s);

          op_s->op = gst_tensor_transform_get_operator (str_op[0]);

          switch (op_s->op) {
            case GTT_OP_TYPECAST:
              if (num_op > 1 && str_op[1]) {
                if (i > 0) {
                  GST_WARNING_OBJECT (filter,
                      "To prevent memory re-allocation, tensor-transform limits the typecast during the sequence. "
                      "Please set the typecast in the first.");
                  op_s->op = GTT_OP_UNKNOWN;
                  break;
                }

                op_s->value.type = get_tensor_type (str_op[1]);

                if (op_s->value.type == _NNS_END) {
                  GST_WARNING_OBJECT (filter, "Unknown tensor type %s",
                      str_op[1]);
                  op_s->op = GTT_OP_UNKNOWN;
                } else {
                  filter->data_arithmetic.out_type = op_s->value.type;
                }
              } else {
                GST_WARNING_OBJECT (filter, "Invalid option for typecast %s",
                    str_operators[i]);
                op_s->op = GTT_OP_UNKNOWN;
              }
              break;
            case GTT_OP_ADD:
            case GTT_OP_MUL:
            case GTT_OP_DIV:
              if (num_op > 1 && str_op[1]) {
                /* get operand */
                if (strchr (str_op[1], '.') || strchr (str_op[1], 'e') ||
                    strchr (str_op[1], 'E')) {
                  double val;

                  val = g_ascii_strtod (str_op[1], NULL);
                  gst_tensor_transform_set_value (filter, &op_s->value,
                      _NNS_FLOAT64, &val);
                } else {
                  int64_t val;

                  val = g_ascii_strtoll (str_op[1], NULL, 10);
                  gst_tensor_transform_set_value (filter, &op_s->value,
                      _NNS_INT64, &val);
                }
              } else {
                GST_WARNING_OBJECT (filter, "Invalid option for arithmetic %s",
                    str_operators[i]);
                op_s->op = GTT_OP_UNKNOWN;
              }
              break;
            default:
              GST_WARNING_OBJECT (filter, "Unknown operator %s", str_op[0]);
              break;
          }

          /* append operator */
          if (op_s->op != GTT_OP_UNKNOWN) {
            filter->operators = g_slist_append (filter->operators, op_s);
          } else {
            g_free (op_s);
          }
        } else {
          GST_WARNING_OBJECT (filter, "Invalid option %s", str_operators[i]);
        }

        g_strfreev (str_op);
      }

      filter->loaded = (filter->operators != NULL);
      g_strfreev (str_operators);
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
      GST_ERROR_OBJECT (filter, "Cannot identify mode\n");
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
    case PROP_ACCELERATION:
#ifdef HAVE_ORC
      filter->acceleration = g_value_get_boolean (value);
      silent_debug ("acceleration = %d\n", filter->acceleration);
#else
      GST_WARNING_OBJECT (filter, "Orc acceleration is not supported");
      filter->acceleration = FALSE;
#endif
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
    case PROP_ACCELERATION:
      g_value_set_boolean (value, filter->acceleration);
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

  if (filter->operators) {
    g_slist_free_full (filter->operators, g_free);
    filter->operators = NULL;
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
  uint32_t *fromDim = filter->in_config.info.dimension;
  uint32_t *toDim = filter->out_config.info.dimension;
  tensor_type in_tensor_type = filter->in_config.info.type;
  int from = filter->data_dimchg.from;
  int to = filter->data_dimchg.to;
  int i, j, k;
  unsigned int loopLimit = 1;
  size_t loopBlockSize = tensor_element_size[in_tensor_type];
  size_t copyblocksize = tensor_element_size[in_tensor_type];
  size_t copyblocklimit = 1;

  if (from == to) {
    /** Useless memcpy. Do not call this or @todo do "IP" operation */
    nns_memcpy (outptr, inptr,
        gst_tensor_info_get_size (&filter->in_config.info));
    GST_WARNING_OBJECT (filter,
        "Calling tensor_transform with high memcpy overhead WITHOUT any effects! Check your stream wheter you really need tensor_transform.\n");
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
          nns_memcpy (j_destptr + copyblocksize * k,
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
  size_t num = get_tensor_element_count (filter->in_config.info.dimension);
  tensor_type in_tensor_type = filter->in_config.info.type;
  tensor_type out_tensor_type = filter->out_config.info.type;

  tensor_transform_operand_s value;
  size_t i, data_idx;

#ifdef HAVE_ORC
  if (orc_supported (filter)) {
    orc_typecast (inptr, outptr, num, in_tensor_type, out_tensor_type);
    return GST_FLOW_OK;
  }
#endif

  for (i = 0; i < num; ++i) {
    /* init value with input tensor type */
    data_idx = tensor_element_size[in_tensor_type] * i;
    gst_tensor_transform_set_value (filter, &value, in_tensor_type,
        (gpointer) (inptr + data_idx));

    /* typecast */
    gst_tensor_transform_typecast_value (filter, &value, out_tensor_type);

    /* set output value */
    g_assert (out_tensor_type == value.type);
    data_idx = tensor_element_size[out_tensor_type] * i;
    gst_tensor_transform_get_value (filter, &value,
        (gpointer) (outptr + data_idx));
  }

  return GST_FLOW_OK;
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
  size_t num = get_tensor_element_count (filter->in_config.info.dimension);
  tensor_type in_tensor_type = filter->in_config.info.type;
  tensor_type out_tensor_type = filter->out_config.info.type;

  GSList *walk;
  tensor_transform_operator_s *op_s;
  tensor_transform_operand_s value;
  size_t i, data_idx;

#ifdef HAVE_ORC
  if (orc_supported (filter)) {
    uint8_t *srcptr = (uint8_t *) inptr;

    walk = filter->operators;
    op_s = (tensor_transform_operator_s *) walk->data;

    if (op_s->op == GTT_OP_TYPECAST) {
      /**
       * Typecast should be called at the first.
       * Do the typecast. If in/out type is same, this will copy the input array to output.
       */
      orc_typecast (inptr, outptr, num, in_tensor_type, out_tensor_type);
      srcptr = outptr;

      walk = g_slist_next (walk);
    }

    while (walk) {
      op_s = (tensor_transform_operator_s *) walk->data;

      if (op_s->op != GTT_OP_TYPECAST) {
        gst_tensor_transform_typecast_value (filter, &op_s->value,
            out_tensor_type);
        orc_operator (srcptr, outptr, num, &op_s->value, op_s->op);
        srcptr = outptr;
      }

      walk = g_slist_next (walk);
    }

    return GST_FLOW_OK;
  }
#endif

  for (i = 0; i < num; ++i) {
    /* init value with input tensor type */
    data_idx = tensor_element_size[in_tensor_type] * i;
    gst_tensor_transform_set_value (filter, &value, in_tensor_type,
        (gpointer) (inptr + data_idx));

    walk = filter->operators;
    while (walk) {
      op_s = (tensor_transform_operator_s *) walk->data;

      /**
       * @todo add more options
       */
      switch (op_s->op) {
        case GTT_OP_TYPECAST:
          gst_tensor_transform_typecast_value (filter, &value,
              op_s->value.type);
          break;
        case GTT_OP_ADD:
        case GTT_OP_MUL:
        case GTT_OP_DIV:
          gst_tensor_transform_typecast_value (filter, &op_s->value,
              value.type);
          gst_tensor_transform_do_operator (filter, &value, &op_s->value,
              op_s->op);
          break;
        default:
          g_assert (0);
          return GST_FLOW_ERROR;
      }

      walk = g_slist_next (walk);
    }

    /* set output value */
    g_assert (out_tensor_type == value.type);
    data_idx = tensor_element_size[out_tensor_type] * i;
    gst_tensor_transform_get_value (filter, &value,
        (gpointer) (outptr + data_idx));
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
	    nns_memcpy(_out, _in, typesize); \
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
  uint32_t *fromDim = filter->in_config.info.dimension;
  tensor_type in_tensor_type = filter->in_config.info.type;
  size_t type_size = tensor_element_size[in_tensor_type];
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
    nns_memcpy (outptr, inptr,
        gst_tensor_info_get_size (&filter->in_config.info));
    GST_WARNING_OBJECT (filter,
        "Calling tensor_transform with high memcpy overhead WITHOUT any effects!");
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
  uint32_t *fromDim = filter->in_config.info.dimension;
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
      GST_ERROR_OBJECT (filter, "Cannot identify mode\n");
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

  g_assert (filter->loaded);
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
 * @param[in] filter "this" pointer
 * @param[in] caps The input caps to be read
 * @param[out] config configured tensor info
 * @return TRUE if successful (both dim/type read). FALSE if not.
 */
static gboolean
gst_tensor_transform_read_caps (GstTensorTransform * filter,
    const GstCaps * caps, GstTensorConfig * config)
{
  GstStructure *structure;

  g_return_val_if_fail (config != NULL, FALSE);

  structure = gst_caps_get_structure (caps, 0);

  if (!gst_structure_has_name (structure, "other/tensor")) {
    GST_WARNING_OBJECT (filter, "caps is not tensor %s\n",
        gst_structure_get_name (structure));
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
  int i;

  switch (filter->mode) {
    case GTT_DIMCHG:
      out_info->type = in_info->type;

      if (direction == GST_PAD_SINK) {
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
      /** For both directions, dimension does not change */
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        out_info->dimension[i] = in_info->dimension[i];
      }
      if (direction == GST_PAD_SINK) {
          /** src = SINKPAD / dest = SRCPAD */
        out_info->type = filter->data_typecast.to;
      } else {
          /** src = SRCPAD / dest = SINKPAD */
        out_info->type = in_info->type;   /** @todo this may cause problems with Cap-Transform */
      }
      break;

    case GTT_ARITHMETIC:
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        out_info->dimension[i] = in_info->dimension[i];
      }
      out_info->type = in_info->type;

      /* check arith mode option has typecast operator */
      if (direction == GST_PAD_SINK &&
          filter->data_arithmetic.out_type != _NNS_END) {
        out_info->type = filter->data_arithmetic.out_type;
      }
      break;

    case GTT_TRANSPOSE:
      out_info->type = in_info->type;

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

    case GTT_STAND:
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        out_info->dimension[i] = in_info->dimension[i];
      }
      out_info->type = in_info->type;
      break;

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

  silent_debug ("Calling TransformCaps, direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filtercap, "filter");

  gst_tensor_config_init (&in_config);
  gst_tensor_config_init (&out_config);

  if (gst_tensor_transform_read_caps (filter, caps, &in_config)) {
    gst_tensor_transform_convert_dimension (filter, direction,
        &in_config.info, &out_config.info);
  }

  /**
   * supposed same framerate from input configuration
   */
  out_config.rate_n = in_config.rate_n;
  out_config.rate_d = in_config.rate_d;

  result = gst_tensor_caps_from_config (&out_config);

  if (filtercap && gst_caps_get_size (filtercap) > 0) {
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

  result =
      gst_tensor_transform_transform_caps (trans, direction, caps, othercaps);
  gst_caps_unref (othercaps);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

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
  GstTensorConfig in_config, out_config;
  GstTensorConfig config;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug ("Calling SetCaps\n");
  silent_debug_caps (incaps, "incaps");
  silent_debug_caps (outcaps, "outcaps");

  if (!gst_tensor_transform_read_caps (filter, incaps, &in_config) ||
      !gst_tensor_config_validate (&in_config)) {
    GST_ERROR_OBJECT (filter, "Cannot read cap of incaps\n");
    goto error;
  }

  if (!gst_tensor_transform_read_caps (filter, outcaps, &out_config) ||
      !gst_tensor_config_validate (&out_config)) {
    GST_ERROR_OBJECT (filter, "Cannot read cap of outcaps\n");
    goto error;
  }

  /* check framerate */
  if (in_config.rate_n != out_config.rate_n
      || in_config.rate_d != out_config.rate_d) {
    GST_ERROR_OBJECT (filter, "Framerate is not matched\n");
    goto error;
  }

  /* compare type and dimension */
  if (!gst_tensor_transform_convert_dimension (filter, GST_PAD_SINK,
          &in_config.info, &config.info) ||
      !gst_tensor_info_is_equal (&out_config.info, &config.info)) {
    GST_ERROR_OBJECT (filter,
        "Tensor info is not matched with given properties.\n");
    goto error;
  }

  /* set in/out tensor info */
  filter->in_config = in_config;
  filter->out_config = out_config;

#ifdef HAVE_ORC
  /**
   * @todo support 64bit integer and remove the flag orc_supported
   */
  if (in_config.info.type != _NNS_INT64 &&
      in_config.info.type != _NNS_UINT64 &&
      out_config.info.type != _NNS_INT64 &&
      out_config.info.type != _NNS_UINT64) {
    filter->orc_supported = TRUE;
  }

  if (orc_supported (filter)) {
    GST_INFO_OBJECT (filter, "Orc acceleration enabled.");
  }
#endif
  return TRUE;
error:
  GST_ERROR_OBJECT (filter, "Set Caps Failed!\n");
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

  /**
   * supposed output tensor configured, then get size from output tensor info.
   */
  *othersize = gst_tensor_info_get_size (&filter->out_config.info);
  return TRUE;
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
