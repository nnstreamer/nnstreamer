/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	gsttensor_transform.c
 * @date	10 Jul 2018
 * @brief	GStreamer plugin to transform tensor dimension or type
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		This is NYI.
 *
 */

/**
 * SECTION:element-tensor_transform
 *
 * A filter that transforms tensors dimension or type.
 * The input and output is always in the format of other/tensor or other/tensors.
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
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "gsttensor_transform.h"

#ifdef HAVE_ORC
#include "nnstreamer-orc.h"
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!filter->silent)
#endif

GST_DEBUG_CATEGORY_STATIC (gst_tensor_transform_debug);
#define GST_CAT_DEFAULT gst_tensor_transform_debug
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_MAKE ("{ static, flexible }")
#define REGEX_DIMCHG_OPTION "^([0-3]):([0-3])$"
#define REGEX_TYPECAST_OPTION "(^[u]?int(8|16|32|64)$|^float(16|32|64)$)"
#define REGEX_TRANSPOSE_OPTION "^(?:([0-2]):(?!.*\\1)){3}3$"
#define REGEX_STAND_OPTION "^(default|dc-average)(:([u]?int(8|16|32|64)|float(16|32|64)))?(,per-channel:(true|false))?$"
#define REGEX_CLAMP_OPTION "^((([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?))):"\
    "((([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)))$"
#define REGEX_ARITH_OPTION "^(typecast:([u]?int(8|16|32|64)|float(16|32|64)),)?"\
    "(per-channel:(false|true@[0-9]+),)?"\
    "(((add|mul|div)(:([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?))+(@[0-9]+)?)(,|))+$"

#define REGEX_ARITH_OPTION_TYPECAST "(typecast:([u]?int(8|16|32|64)|float(16|32|64)))"

/**
 * @brief The transpose rank is fixed to 4.
 * This RANK does not affect other/tensors(s)'s NNS_TENSOR_RANK_LIMIT.
 */
#define NNS_TENSOR_TRANSPOSE_RANK_LIMIT (4)

/**
 * @brief tensor_transform properties
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_MODE,
  PROP_OPTION,
  PROP_ACCELERATION,
  PROP_APPLY,
  PROP_TRANSPOSE_RANK_LIMIT
};

/**
 * @brief Flag to set orc acceleration.
 */
#ifdef HAVE_ORC
#define DEFAULT_ACCELERATION TRUE
#else
#define DEFAULT_ACCELERATION FALSE
#endif

static const gchar *gst_tensor_transform_stand_string[] = {
  [STAND_DEFAULT] = "default",
  [STAND_DC_AVERAGE] = "dc-average",
  [STAND_END] = NULL
};

static const gchar *gst_tensor_transform_operator_string[] = {
  [GTT_OP_TYPECAST] = "typecast",
  [GTT_OP_ADD] = "add",
  [GTT_OP_MUL] = "mul",
  [GTT_OP_DIV] = "div",
  [GTT_OP_UNKNOWN] = NULL
};

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

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

static gboolean gst_tensor_transform_convert_dimension (GstTensorTransform *
    filter, GstPadDirection direction, guint idx, const GstTensorInfo * in_info,
    GstTensorInfo * out_info);

#define GST_TYPE_TENSOR_TRANSFORM_MODE (gst_tensor_transform_mode_get_type ())
/**
 * @brief A private function to register GEnumValue array for the 'mode' property
 *        to a GType and return it
 */
static GType
gst_tensor_transform_mode_get_type (void)
{
  static GType mode_type = 0;

  if (mode_type == 0) {
    static GEnumValue mode_types[] = {
      {GTT_DIMCHG, "Mode for changing tensor dimensions, "
            "option=FROM_DIM:TO_DIM (with a regex, " REGEX_DIMCHG_OPTION
            ", where NNS_TENSOR_RANK_LIMIT is 4)",
          "dimchg"},
      {GTT_TYPECAST, "Mode for casting type of tensor, "
            "option=" REGEX_TYPECAST_OPTION, "typecast"},
      {GTT_ARITHMETIC, "Mode for arithmetic operations with tensor, "
            "option=[typecast:TYPE,][per-channel:(false|true@DIM),]add|mul|div:NUMBER[@CH_IDX], ...",
          "arithmetic"},
      {GTT_TRANSPOSE, "Mode for transposing shape of tensor, "
            "option=D1\':D2\':D3\':D4 (fixed to 3)",
          "transpose"},
      {GTT_STAND, "Mode for statistical standardization of tensor, "
            "option=(default|dc-average)[:TYPE][,per-channel:(false|true)]",
          "stand"},
      {GTT_CLAMP, "Mode for clamping all elements of tensor into the range, "
            "option=CLAMP_MIN:CLAMP_MAX",
          "clamp"},
      {GTT_UNKNOWN, "Unknown or not-implemented-yet mode",
          "unknown"},
      {0, NULL, NULL},
    };

    mode_type = g_enum_register_static ("gtt_mode_type", mode_types);
  }

  return mode_type;
}

/**
 * @brief initialize the tensor_transform's class
 */
static void
gst_tensor_transform_class_init (GstTensorTransformClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_transform_debug, "tensor_transform", 0,
      "Element to transforms tensor dimension or type");

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
      g_param_spec_enum ("mode", "Mode", "Mode used for transforming tensor",
          GST_TYPE_TENSOR_TRANSFORM_MODE, GTT_UNKNOWN,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OPTION,
      g_param_spec_string ("option", "Option",
          "Option for the tensor transform mode ?", "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_ACCELERATION,
      g_param_spec_boolean ("acceleration", "Acceleration", "Orc acceleration",
          DEFAULT_ACCELERATION, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_APPLY,
      g_param_spec_string ("apply", "Apply", "Select tensors to apply, "
          "separated with ',' in case of multiple tensors. Default to apply all tensors.",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TRANSPOSE_RANK_LIMIT,
      g_param_spec_uint ("transpose-rank-limit", "Transpose rank limit",
          "The rank limit of transpose, which varies per version of nnstreamer and may be lower than the global rank limit if it is over 4.",
          0, NNS_TENSOR_RANK_LIMIT, NNS_TENSOR_TRANSPOSE_RANK_LIMIT,
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorTransform",
      "Filter/Tensor",
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
  filter->mode = GTT_UNKNOWN;
  filter->option = NULL;
  filter->loaded = FALSE;
  filter->operators = NULL;
  filter->acceleration = DEFAULT_ACCELERATION;
  filter->apply = NULL;

  gst_tensors_config_init (&filter->in_config);
  gst_tensors_config_init (&filter->out_config);
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

#ifndef FLOAT16_SUPPORT
/**
 * @brief Generate error if float16 is required.
 */
static void
float16_not_supported (void)
{
  ml_loge
      ("Tensor_tranform does not support float16 operators. Apply -Denable-float16=true for meson build option if your architecture support float16. Note that tensor-transform's float16 is adhoc and does NOT perform good (slow!).\n");
  g_assert (0);
}
#endif

#ifdef FLOAT16_SUPPORT
/**
 * @brief Refrain from heavy operations on float16
 * @todo Remove this after applying SIMD or ORC
 * */
static void
refrain_from_heavy_op_on_float16 (gulong n)
{
  static int warned = 0;
  /* 1 million */
  if (n > 1000000) {
    if (warned)
      return;
    ml_logw
        ("Tensor_transform implementation for float16 does not support SIMD. Heavy tensor-transform operations of float16 is not recommended. Try to apply heavy ops with other types (e.g., float32) and convert it to float16 at the time when it's really needed.\n");
    warned = 1;
  }
}

/** @todo Make this use SIMD or ORC */
#define _conv_to_f16(intype, o, i, n) \
  do { \
    float16 *op = (gpointer) (o); \
    intype *ip = (gpointer) (i); \
    gulong idx; \
    refrain_from_heavy_op_on_float16 (n); \
    for (idx = 0; idx < n; idx++) \
      *(op + idx) = (float16) *(ip + idx); \
  } while (0)

/** @todo Make this use SIMD or ORC */
#define _conv_from_f16_action(n, op, ip, otypename) \
  do { \
    gulong idx; \
    for (idx = 0; idx < n; idx++) \
      *(op + idx) = (otypename) *(ip + idx); \
  } while (0)

/** @todo Make this use SIMD or ORC */
#define _conv_from_f16(otype, o, i, n) \
  do { \
    float16 *ip = (gpointer) (i); \
    refrain_from_heavy_op_on_float16 (n); \
    switch (otype) { \
      case _NNS_INT32: { \
        int32_t *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, int32_t); \
        break; } \
      case _NNS_UINT32: {  \
        uint32_t *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, uint32_t); \
        break; } \
      case _NNS_INT16: {  \
        int16_t *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, int16_t); \
        break; } \
      case _NNS_UINT16: {  \
        uint16_t *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, uint16_t); \
        break; } \
      case _NNS_INT8: {  \
        int8_t *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, int8_t); \
        break; } \
      case _NNS_UINT8: {  \
        uint8_t *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, uint8_t); \
        break; } \
      case _NNS_FLOAT64: {  \
        double *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, double); \
        break; } \
      case _NNS_FLOAT32: {  \
        float *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, float); \
        break; } \
      case _NNS_FLOAT16: {  \
        float16 *op = (gpointer) (o); \
        _conv_from_f16_action (n, op, ip, float16); \
        break; } \
      default: GST_ERROR_OBJECT (filter, "Unsupported type %d", (otype)); g_assert (0); \
    } \
  } while (0)

/** @todo Make this use SIMD or ORC */
#define _op_float16(i, n, v, op) \
  do { \
    gulong idx; \
    float16 *data_in = (float16 *) (i); \
    refrain_from_heavy_op_on_float16 (n); \
    switch (op) { \
      case GTT_OP_ADD: \
        for (idx = 0; idx < n; idx++) \
          data_in[idx] = data_in[idx] + (v); \
        break; \
      case GTT_OP_MUL: \
        for (idx = 0; idx < n; idx++) \
          data_in[idx] = data_in[idx] * (v); \
        break; \
      case GTT_OP_DIV: \
        for (idx = 0; idx < n; idx++) \
          data_in[idx] = data_in[idx] / (v); \
        break; \
      default: GST_ERROR_OBJECT (filter, "Unknown operator for float16: %d", op); break; \
    } \
  } while (0)

#else /* ! FLOAT16_SUPPORT */
#define _conv_to_f16(intype, o, i, n) do { float16_not_supported (); } while (0)
#define _conv_from_f16(otype, o, i, n) do { float16_not_supported (); } while (0)
#define _op_float16(i, n, v, op) do { float16_not_supported (); } while (0)
#endif /* FLOAT16_SUPPORT */

#ifdef HAVE_ORC
/* define macros for orc */
/** @todo support 64bit integer and remove below line */
#define type_64bit_integer(t) ((t) == _NNS_INT64 || (t) == _NNS_UINT64)
#define orc_supported(f,itype,otype) ((f)->acceleration && !(type_64bit_integer (itype) || type_64bit_integer (otype)))

#define orc_func_conv(intype,outtype) nns_orc_conv_ ## intype ## _to_ ## outtype
#define orc_func_add(intype) nns_orc_add_c_ ## intype
#define orc_func_mul(intype) nns_orc_mul_c_ ## intype
#define orc_func_div(intype) nns_orc_div_c_ ## intype

#define orc_typecast_to(i,o,n,intype,otype,intypename) do { \
    switch (otype) { \
      case _NNS_INT32: orc_func_conv (intype, s32) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_UINT32: orc_func_conv (intype, u32) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_INT16: orc_func_conv (intype, s16) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_UINT16: orc_func_conv (intype, u16) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_INT8: orc_func_conv (intype, s8) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_UINT8: orc_func_conv (intype, u8) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_FLOAT64: orc_func_conv (intype, f64) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_FLOAT32: orc_func_conv (intype, f32) ((gpointer) o, (gpointer) i, n); break; \
      case _NNS_FLOAT16: _conv_to_f16 (intypename, o, i, n); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported output type %d", otype); g_assert (0); break; \
    } \
  } while (0)

#define orc_typecast(i,o,n,itype,otype) do { \
    switch (itype) { \
      case _NNS_INT32: orc_typecast_to (i, o, n, s32, otype, int32_t); break; \
      case _NNS_UINT32: orc_typecast_to (i, o, n, u32, otype, uint32_t); break; \
      case _NNS_INT16: orc_typecast_to (i, o, n, s16, otype, int16_t); break; \
      case _NNS_UINT16: orc_typecast_to (i, o, n, u16, otype, uint16_t); break; \
      case _NNS_INT8: orc_typecast_to (i, o, n, s8, otype, int8_t); break; \
      case _NNS_UINT8: orc_typecast_to (i, o, n, u8, otype, uint8_t); break; \
      case _NNS_FLOAT64: orc_typecast_to (i, o, n, f64, otype, double); break; \
      case _NNS_FLOAT32: orc_typecast_to (i, o, n, f32, otype, float); break; \
      case _NNS_FLOAT16: _conv_from_f16 (otype, o, i, n); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported input type %d", itype); g_assert (0); break; \
    } \
  } while (0)

#define orc_operator_func(i,n,v,opfunc,op) do { \
    switch ((v)->type) { \
      case _NNS_INT32: opfunc (s32) ((gpointer) i, (v)->data._int32_t, n); break; \
      case _NNS_UINT32: opfunc (u32) ((gpointer) i, (v)->data._uint32_t, n); break; \
      case _NNS_INT16: opfunc (s16) ((gpointer) i, (v)->data._int16_t, n); break; \
      case _NNS_UINT16: opfunc (u16) ((gpointer) i, (v)->data._uint16_t, n); break; \
      case _NNS_INT8: opfunc (s8) ((gpointer) i, (v)->data._int8_t, n); break; \
      case _NNS_UINT8: opfunc (u8) ((gpointer) i, (v)->data._uint8_t, n); break; \
      case _NNS_FLOAT64: opfunc (f64) ((gpointer) i, (v)->data._double, n); break; \
      case _NNS_FLOAT32: opfunc (f32) ((gpointer) i, (v)->data._float, n); break; \
      case _NNS_FLOAT16: _op_float16 (i, n, (v)->data._float16, op); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported type %d", (v)->type); g_assert (0); break; \
    } \
  } while (0)

#define orc_operator_div_loop(i,n,val,typename) do { \
    gsize idx_div; \
    typename *data_in = (typename *) (i); \
    for (idx_div = 0; idx_div < (n); ++idx_div) { \
      data_in[idx_div] = data_in[idx_div] / (val); \
    } \
  } while (0)

#define orc_operator(i,n,v,op) do { \
    switch (op) { \
      case GTT_OP_ADD: orc_operator_func (i, n, v, orc_func_add, op); break; \
      case GTT_OP_MUL: orc_operator_func (i, n, v, orc_func_mul, op); break; \
      case GTT_OP_DIV: \
        switch ((v)->type) { \
          case _NNS_INT32: orc_operator_div_loop (i, n, (v)->data._int32_t, int32_t); break; \
          case _NNS_UINT32: orc_operator_div_loop (i, n, (v)->data._uint32_t, uint32_t); break; \
          case _NNS_INT16: orc_operator_div_loop (i, n, (v)->data._int16_t, int16_t); break; \
          case _NNS_UINT16: orc_operator_div_loop (i, n, (v)->data._uint16_t, uint16_t); break; \
          case _NNS_INT8: orc_operator_div_loop (i, n, (v)->data._int8_t, int8_t); break; \
          case _NNS_UINT8: orc_operator_div_loop (i, n, (v)->data._uint8_t, uint8_t); break; \
          case _NNS_FLOAT64: orc_func_div (f64) ((gpointer) i, (v)->data._double, n); break; \
          case _NNS_FLOAT32: orc_func_div (f32) ((gpointer) i, (v)->data._float, n); break; \
          case _NNS_FLOAT16: _op_float16 (i, n, (v)->data._float16, op); break; \
          default: GST_ERROR_OBJECT (filter, "Unsupported type %d", (v)->type); g_assert (0); break; \
        } \
        break; \
      default: GST_ERROR_OBJECT (filter, "Unknown operator %d", op); break; \
    } \
  } while (0)
#endif /* HAVE_ORC */

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
    tensor_data_s * desc, const tensor_data_s * val,
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
    case _NNS_FLOAT16:
#ifdef FLOAT16_SUPPORT
      handle_operator (desc, val, op, float16);
#else
      float16_not_supported ();
#endif
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
 * @brief Setup internal data (data_* in GstTensorTransform)
 * @param[in/out] filter "this" pointer. mode & option MUST BE set already.
 * @retval TRUE if OK or operation-skipped, FALSE if fatal-error.
 */
static gboolean
gst_tensor_transform_set_option_data (GstTensorTransform * filter)
{
  gchar *filter_name;
  gboolean ret = FALSE;

  if (filter->mode == GTT_UNKNOWN || filter->option == NULL)
    return TRUE;

  filter_name = gst_object_get_name ((GstObject *) filter);

  switch (filter->mode) {
    case GTT_DIMCHG:
    {
      gchar **strv = NULL;

      if (!g_regex_match_simple (REGEX_DIMCHG_OPTION, filter->option,
              G_REGEX_CASELESS, 0)) {
        ml_loge
            ("%s: dimchg: \'%s\' is not valid option string: it should be in the form of IDX_DIM_FROM:IDX_DIM_TO: with a regex, "
            REGEX_DIMCHG_OPTION "\n", filter_name, filter->option);
        break;
      }

      strv = g_strsplit (filter->option, ":", 2);

      filter->data_dimchg.from = (int) g_ascii_strtoll (strv[0], NULL, 10);
      filter->data_dimchg.to = (int) g_ascii_strtoll (strv[1], NULL, 10);
      ret = filter->loaded = TRUE;
      g_strfreev (strv);
      break;
    }
    case GTT_TYPECAST:
    {
      if (g_regex_match_simple (REGEX_TYPECAST_OPTION, filter->option,
              G_REGEX_CASELESS, 0)) {
        filter->data_typecast.to = gst_tensor_get_type (filter->option);
        ret = filter->loaded = TRUE;
      } else {
        ml_loge
            ("%s: typecast: \'%s\' is not valid data type for tensor: data type of tensor should be one of %s\n",
            filter_name, filter->option, GST_TENSOR_TYPE_ALL);
      }
      break;
    }
    case GTT_ARITHMETIC:
    {
      gchar *str_option;
      gchar **str_operators;
      gchar **str_op;
      tensor_transform_operator_s *op_s;
      guint i, num_operators, num_op;
      GRegex *regex_option_tc;

      filter->data_arithmetic.out_type = _NNS_END;
      filter->data_arithmetic.per_channel_arith = FALSE;

      if (filter->operators) {
        GST_WARNING_OBJECT (filter,
            "There exists pre-defined operators (total %d), now reset these.",
            g_slist_length (filter->operators));

        g_slist_free_full (filter->operators, g_free);
        filter->operators = NULL;
      }

      regex_option_tc = g_regex_new (REGEX_ARITH_OPTION_TYPECAST,
          G_REGEX_CASELESS, 0, 0);

      if (!regex_option_tc) {
        GST_ERROR_OBJECT (filter,
            "arithmetic: failed to create a GRegex structure for %s\n",
            REGEX_ARITH_OPTION_TYPECAST);
        break;
      }

      if (g_regex_match_full (regex_option_tc, filter->option, -1,
              1, 0, NULL, NULL)) {
        str_option = g_regex_replace (regex_option_tc, filter->option, -1, 1,
            "", 0, 0);
        ml_loge
            ("%s: arithmetic: [typecast:TYPE,] should be located at the first to prevent memory re-allocation: typecast(s) in the middle of \'%s\' will be ignored\n",
            filter_name, filter->option);
      } else {
        str_option = g_strdup (filter->option);
      }
      g_regex_unref (regex_option_tc);

      if (!g_regex_match_simple (REGEX_ARITH_OPTION, str_option,
              G_REGEX_CASELESS, 0)) {
        ml_loge
            ("%s: arithmetic: \'%s\' is not valid option string: it should be in the form of [typecast:TYPE,][per-channel:(false|true@DIM),]add|mul|div:NUMBER[@CH_IDX]..., ...\n",
            filter_name, str_option);
        g_free (str_option);
        break;
      }
      str_operators = g_strsplit (str_option, ",", -1);
      num_operators = g_strv_length (str_operators);

      for (i = 0; i < num_operators; ++i) {
        str_op = g_strsplit (str_operators[i], ":", -1);
        num_op = g_strv_length (str_op);

        if (str_op[0]) {
          gchar **values = g_strsplit (str_op[1], "@", -1);
          guint num_values = g_strv_length (values);

          /* check whether per-channel */
          if (g_ascii_strcasecmp (str_op[0], "per-channel") == 0) {
            if (num_values > 1 && g_ascii_strcasecmp (values[0], "true") == 0) {
              ml_logi
                  ("Set per-channel for arithmetic and assume that %s-th dim is the channel",
                  values[1]);
              filter->data_arithmetic.per_channel_arith = TRUE;
              filter->data_arithmetic.ch_dim =
                  g_ascii_strtoull (values[1], NULL, 10);
            }

            g_strfreev (values);
            g_strfreev (str_op);
            continue;
          }

          op_s = g_new0 (tensor_transform_operator_s, 1);
          g_assert (op_s);

          op_s->op = gst_tensor_transform_get_operator (str_op[0]);
          op_s->applying_ch = -1;       /* -1 means applying to all channels */
          switch (op_s->op) {
            case GTT_OP_TYPECAST:
              if (num_op > 1 && str_op[1]) {
                op_s->value.type = gst_tensor_get_type (values[0]);
                filter->data_arithmetic.out_type = op_s->value.type;
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
                if (strchr (values[0], '.') || strchr (values[0], 'e') ||
                    strchr (values[0], 'E')) {
                  double val;

                  val = g_ascii_strtod (values[0], NULL);
                  gst_tensor_data_set (&op_s->value, _NNS_FLOAT64, &val);
                } else {
                  int64_t val;

                  val = g_ascii_strtoll (values[0], NULL, 10);
                  gst_tensor_data_set (&op_s->value, _NNS_INT64, &val);
                }

                if (filter->data_arithmetic.per_channel_arith && num_values > 1) {
                  op_s->applying_ch = g_ascii_strtoll (values[1], NULL, 10);
                }

              } else {
                GST_WARNING_OBJECT (filter,
                    "Invalid option for arithmetic %s", str_operators[i]);
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

          g_strfreev (values);
        } else {
          GST_WARNING_OBJECT (filter, "Invalid option %s", str_operators[i]);
        }

        g_strfreev (str_op);
      }

      ret = filter->loaded = (filter->operators != NULL);
      g_strfreev (str_operators);
      g_free (str_option);
      break;
    }
    case GTT_TRANSPOSE:
    {
      int i;
      gchar **strv = NULL;

      if (!g_regex_match_simple (REGEX_TRANSPOSE_OPTION, filter->option,
              G_REGEX_CASELESS, 0)) {
        ml_loge
            ("%s: transpose: \'%s\' is not valid option string: it should be in the form of NEW_IDX_DIM0:NEW_IDX_DIM1:NEW_IDX_DIM2:3 (Now transpose mode's rank is fixed to 3. Note that the index of the last dim is always fixed to 3)\n",
            filter_name, filter->option);
        break;
      }

      strv = g_strsplit (filter->option, ":", NNS_TENSOR_TRANSPOSE_RANK_LIMIT);
      for (i = 0; i < NNS_TENSOR_TRANSPOSE_RANK_LIMIT; i++) {
        filter->data_transpose.trans_order[i] =
            (uint8_t) g_ascii_strtoull (strv[i], NULL, 10);
      }

      ret = filter->loaded = TRUE;
      g_strfreev (strv);
      break;
    }
    case GTT_STAND:
    {
      gchar **options = NULL;
      guint i, num_options;

      if (!g_regex_match_simple (REGEX_STAND_OPTION, filter->option,
              G_REGEX_CASELESS, 0)) {
        ml_loge
            ("%s: stand: \'%s\' is not a valid option string: it should be in the form of (default|dc-average)[:TYPE][,per-channel:(false|true)]\n",
            filter_name, filter->option);
        break;
      }

      filter->data_stand.out_type = _NNS_END;
      filter->data_stand.per_channel = FALSE;

      options = g_strsplit (filter->option, ",", -1);
      num_options = g_strv_length (options);

      for (i = 0; i < num_options; i++) {
        gchar **strv = g_strsplit (options[i], ":", -1);

        if (g_ascii_strcasecmp (strv[0], "default") == 0 ||
            g_ascii_strcasecmp (strv[0], "dc-average") == 0) {
          filter->data_stand.mode =
              gst_tensor_transform_get_stand_mode (strv[0]);
          if (g_strv_length (strv) > 1)
            filter->data_stand.out_type = gst_tensor_get_type (strv[1]);
        } else if (g_ascii_strcasecmp (strv[0], "per-channel") == 0) {
          if (g_strv_length (strv) > 1 &&
              g_ascii_strcasecmp (strv[1], "true") == 0)
            filter->data_stand.per_channel = TRUE;
        } else {
          filter->data_stand.mode = STAND_END;
          ml_logw ("Unknown option for stand mode: %s", strv[0]);
        }

        g_strfreev (strv);
      }

      g_strfreev (options);
      ret = filter->loaded = TRUE;
      break;
    }
    case GTT_CLAMP:
    {
      gchar **strv = NULL;

      if (!g_regex_match_simple (REGEX_CLAMP_OPTION, filter->option,
              G_REGEX_CASELESS, 0)) {
        ml_loge
            ("%s: clamp: \'%s\' is not valid option string: it should be in the form of [CLAMP_MIN:CLAMP_MAX]\n",
            filter_name, filter->option);
        break;
      }

      strv = g_strsplit (filter->option, ":", 2);

      filter->data_clamp.min = g_ascii_strtod (strv[0], NULL);
      if (errno == ERANGE) {
        ml_loge ("%s: clamp: CLAMP_MIN value has an invalid range\n",
            filter_name);
        g_strfreev (strv);
        break;
      }
      filter->data_clamp.max = g_ascii_strtod (strv[1], NULL);
      if (errno == ERANGE) {
        ml_loge ("%s: clamp: CLAMP_MAX value has an invalid range\n",
            filter_name);
        g_strfreev (strv);
        break;
      }

      g_strfreev (strv);

      if (filter->data_clamp.min > filter->data_clamp.max) {
        ml_loge ("%s: clamp: CLAMP_MIN is larger than CLAMP_MAX\n",
            filter_name);
        break;
      }

      ret = filter->loaded = TRUE;
      break;
    }
    default:
      GST_ERROR_OBJECT (filter, "Cannot identify mode\n");
      ret = FALSE;
  }

  g_free (filter_name);
  return ret;
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
      filter->mode = g_value_get_enum (value);
      gst_tensor_transform_set_option_data (filter);
      break;
    case PROP_OPTION:
    {
      gchar *backup_option = filter->option;
      filter->option = g_value_dup_string (value);
      if (gst_tensor_transform_set_option_data (filter)) {
        silent_debug (filter, "Option = %s --> %s\n", backup_option,
            filter->option);
        g_free (backup_option);
      } else {
        /* ERROR! Revert the change! */
        g_free (filter->option);
        filter->option = backup_option;
        gst_tensor_transform_set_option_data (filter);
      }
      break;
    }
    case PROP_ACCELERATION:
#ifdef HAVE_ORC
      filter->acceleration = g_value_get_boolean (value);
      silent_debug (filter, "acceleration = %d\n", filter->acceleration);
#else
      GST_WARNING_OBJECT (filter, "Orc acceleration is not supported");
      filter->acceleration = FALSE;
#endif
      break;
    case PROP_APPLY:
    {
      gint64 val;
      const gchar *param = g_value_get_string (value);
      gchar **strv = g_strsplit_set (param, ",", -1);
      guint i, num = g_strv_length (strv);
      gchar *endptr = NULL;

      for (i = 0; i < num; i++) {
        errno = 0;
        val = g_ascii_strtoll (strv[i], &endptr, 10);
        if (errno == ERANGE || errno == EINVAL || (endptr == strv[i])) {
          ml_loge ("Cannot convert string %s to a gint64 value", strv[i]);
        }
        filter->apply = g_list_append (filter->apply, GINT_TO_POINTER (val));
      }
      g_strfreev (strv);
      break;
    }
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
      g_value_set_enum (value, filter->mode);
      break;
    case PROP_OPTION:
      g_value_set_string (value, filter->option);
      break;
    case PROP_ACCELERATION:
      g_value_set_boolean (value, filter->acceleration);
      break;
    case PROP_APPLY:
    {
      GList *list;
      gchar *p;
      GPtrArray *arr;
      gchar **strings;

      if (filter->apply == NULL) {
        g_value_set_string (value, "");
        return;
      }

      arr = g_ptr_array_new ();
      for (list = filter->apply; list != NULL; list = list->next) {
        g_ptr_array_add (arr, g_strdup_printf ("%i",
                GPOINTER_TO_INT (list->data)));
      }
      g_ptr_array_add (arr, NULL);
      strings = (gchar **) g_ptr_array_free (arr, FALSE);
      p = g_strjoinv (",", strings);

      g_strfreev (strings);
      g_value_take_string (value, p);
      break;
    }
    case PROP_TRANSPOSE_RANK_LIMIT:
      g_value_set_uint (value, NNS_TENSOR_TRANSPOSE_RANK_LIMIT);
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

  if (filter->apply) {
    g_list_free (filter->apply);
    filter->apply = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief subrouting for tensor-tranform, "dimchg" case.
 * @param[in/out] filter "this" pointer
 * @param[in] in_info input tensor info
 * @param[in] out_info output tensor info
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_dimchg (GstTensorTransform * filter,
    GstTensorInfo * in_info, GstTensorInfo * out_info,
    const uint8_t * inptr, uint8_t * outptr)
{
  uint32_t *fromDim = in_info->dimension;
  uint32_t *toDim = out_info->dimension;
  unsigned int from = filter->data_dimchg.from;
  unsigned int to = filter->data_dimchg.to;
  unsigned int i, j, k;
  unsigned int loopLimit = 1;
  gsize loopBlockSize, copyblocksize, copyblocklimit;

  if (from == to) {
    /** Useless memcpy. Do not call this or @todo do "IP" operation */
    nns_memcpy (outptr, inptr, gst_tensor_info_get_size (in_info));
    GST_WARNING_OBJECT (filter,
        "Calling tensor_transform with high memcpy overhead WITHOUT any effects! Check your stream whether you really need tensor_transform.\n");
    return GST_FLOW_OK;
  }

  g_assert (from < NNS_TENSOR_RANK_LIMIT);
  g_assert (to < NNS_TENSOR_RANK_LIMIT);
  g_assert (fromDim[from] == toDim[to]);

  loopBlockSize = copyblocksize = gst_tensor_get_element_size (in_info->type);
  copyblocklimit = 1;

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
    ml_loge
        ("tensor-transform/dimchg operation is not permitted if from >= to.\n");
    return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * @brief subrouting for tensor-tranform, "typecast" case.
 * @param[in/out] filter "this" pointer
 * @param[in] in_info input tensor info
 * @param[in] out_info output tensor info
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_typecast (GstTensorTransform * filter,
    GstTensorInfo * in_info, GstTensorInfo * out_info,
    const uint8_t * inptr, uint8_t * outptr)
{
  gulong i, num;
  gsize in_element_size, out_element_size;

  num = gst_tensor_get_element_count (in_info->dimension);

#ifdef HAVE_ORC
  if (orc_supported (filter, in_info->type, out_info->type)) {
    orc_typecast (inptr, outptr, num, in_info->type, out_info->type);
    return GST_FLOW_OK;
  }
#endif

  in_element_size = gst_tensor_get_element_size (in_info->type);
  out_element_size = gst_tensor_get_element_size (out_info->type);

  for (i = 0; i < num; ++i) {
    gst_tensor_data_raw_typecast (
        (gpointer) (inptr + in_element_size * i), in_info->type,
        (gpointer) (outptr + out_element_size * i), out_info->type);
  }

  return GST_FLOW_OK;
}

/**
 * @brief subrouting for tensor-tranform, "arithmetic" case.
 * @param[in/out] filter "this" pointer
 * @param[in] in_info input tensor info
 * @param[in] out_info output tensor info
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_arithmetic (GstTensorTransform * filter,
    GstTensorInfo * in_info, GstTensorInfo * out_info,
    const uint8_t * inptr, uint8_t * outptr)
{
  gulong i, num, j, ch;
  gsize in_element_size, out_element_size;

  GSList *walk;
  tensor_transform_operator_s *op_s;
  tensor_data_s value;

  num = gst_tensor_get_element_count (in_info->dimension);

#ifdef HAVE_ORC
  /** per-channel is not supported by orc */
  if (!filter->data_arithmetic.per_channel_arith
      && orc_supported (filter, in_info->type, out_info->type)) {
    walk = filter->operators;

    /**
     * Typecast should be called at the first.
     * Do the typecast. If in/out type is same, this will copy the input array to output.
     */
    orc_typecast (inptr, outptr, num, in_info->type, out_info->type);

    while (walk) {
      op_s = (tensor_transform_operator_s *) walk->data;

      if (op_s->op != GTT_OP_TYPECAST) {
        gst_tensor_data_typecast (&op_s->value, out_info->type);
        orc_operator (outptr, num, &op_s->value, op_s->op);
      }

      walk = g_slist_next (walk);
    }

    return GST_FLOW_OK;
  }
#endif

  in_element_size = gst_tensor_get_element_size (in_info->type);
  out_element_size = gst_tensor_get_element_size (out_info->type);

  /* per-channel */
  if (filter->data_arithmetic.per_channel_arith) {
    guint ch_dim = filter->data_arithmetic.ch_dim;
    gsize ch_offset, ch_size = 1;
    for (i = 0; i < ch_dim; ++i) {
      ch_size *= in_info->dimension[i];
    }
    ch_offset = ch_size * in_info->dimension[ch_dim];

    /** In case of 3:4:4:1,
     * ch_dim:0 -> #ch: 3, ch_size: 1, ch_offset: 3
     * ch_dim:1 -> #ch: 4, ch_size: 3, ch_offset: 12
     * ch_dim:2 -> #ch: 4, ch_size: 12, ch_offset: 48
     * ch_dim:3 -> #ch: 1, ch_size: 48, ch_offset: 48 * 4
     */

    for (i = 0; i < num / ch_offset; ++i) {
      for (ch = 0; ch < in_info->dimension[ch_dim]; ++ch) {
        for (j = 0; j < ch_size; ++j) {
          gulong data_idx = (i * ch_offset) + (ch * ch_size) + j;
          gst_tensor_data_set (&value, in_info->type,
              (gpointer) (inptr + in_element_size * data_idx));

          walk = filter->operators;
          while (walk) {
            op_s = (tensor_transform_operator_s *) walk->data;
            switch (op_s->op) {
              case GTT_OP_TYPECAST:
                gst_tensor_data_typecast (&value, op_s->value.type);
                break;
              case GTT_OP_ADD:
              case GTT_OP_MUL:
              case GTT_OP_DIV:
              {
                gst_tensor_data_typecast (&op_s->value, value.type);

                if (op_s->applying_ch == (int) ch || op_s->applying_ch == -1) {
                  gst_tensor_transform_do_operator (filter, &value,
                      &op_s->value, op_s->op);
                }
                break;
              }
              default:
                g_assert (0);
                return GST_FLOW_ERROR;
            }

            walk = g_slist_next (walk);
          }

          /* set output value */
          g_assert (out_info->type == value.type);
          gst_tensor_data_get (&value, outptr + out_element_size * data_idx);
        }
      }
    }

    return GST_FLOW_OK;
  }

  for (i = 0; i < num; ++i) {
    /* init value with input tensor type */
    gst_tensor_data_set (&value, in_info->type,
        (gpointer) (inptr + in_element_size * i));

    walk = filter->operators;
    while (walk) {
      op_s = (tensor_transform_operator_s *) walk->data;

      /**
       * @todo add more options
       */
      switch (op_s->op) {
        case GTT_OP_TYPECAST:
          gst_tensor_data_typecast (&value, op_s->value.type);
          break;
        case GTT_OP_ADD:
        case GTT_OP_MUL:
        case GTT_OP_DIV:
          gst_tensor_data_typecast (&op_s->value, value.type);
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
    g_assert (out_info->type == value.type);
    gst_tensor_data_get (&value, outptr + out_element_size * i);
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
            const uint8_t *_in; \
            uint8_t *_out; \
            outidx = si*sj*sk*cl + sj*sk*ci + sk*cj + ck; \
            inidx = SK*SJ*SI*l + SJ*SI*k + SI*j + i; \
            _in = inptr + inidx * typesize; \
            _out = outptr + outidx * typesize; \
            nns_memcpy(_out, _in, typesize); \
	  }                                                      \
  } while(0);

/**
 * @brief subrouting for tensor-tranform, "transpose" case.
 * @param[in/out] filter "this" pointer
 * @param[in] in_info input tensor info
 * @param[in] out_info output tensor info
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_transpose (GstTensorTransform * filter,
    GstTensorInfo * in_info, GstTensorInfo * out_info,
    const uint8_t * inptr, uint8_t * outptr)
{
  int i, from, to;
  gboolean checkdim = FALSE;
  uint32_t *fromDim = in_info->dimension;
  gsize type_size = gst_tensor_get_element_size (in_info->type);
  gsize indexI, indexJ, SL, SI, SJ, SK;
  UNUSED (out_info);

  for (i = 0; i < NNS_TENSOR_TRANSPOSE_RANK_LIMIT; i++) {
    from = i;
    to = filter->data_transpose.trans_order[i];
    if (from != to) {
      checkdim = TRUE;
      break;
    }
  }

  if (!checkdim) {
    nns_memcpy (outptr, inptr, gst_tensor_info_get_size (in_info));
    GST_WARNING_OBJECT (filter,
        "Calling tensor_transform with high memcpy overhead WITHOUT any effects!");
    return GST_FLOW_OK;
  }

  indexI = filter->data_transpose.trans_order[0];
  indexJ = filter->data_transpose.trans_order[1];
  SL = fromDim[3], SI = fromDim[0], SJ = fromDim[1], SK = fromDim[2];

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
 * @param[in] in_info input tensor info
 * @param[in] out_info output tensor info
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_stand (GstTensorTransform * filter,
    GstTensorInfo * in_info, GstTensorInfo * out_info,
    const uint8_t * inptr, uint8_t * outptr)
{
  GstFlowReturn ret = GST_FLOW_OK;
  gsize in_element_size, out_element_size, data_size, ch_size;
  gulong i, num, data_idx, ch;
  gdouble tmp, *average, *std;

  in_element_size = gst_tensor_get_element_size (in_info->type);
  out_element_size = gst_tensor_get_element_size (out_info->type);
  num = gst_tensor_get_element_count (in_info->dimension);

  data_size = gst_tensor_info_get_size (in_info);
  ch_size = in_info->dimension[0];

  /* calc average and std */
  average = std = NULL;
  if (filter->data_stand.per_channel) {
    gst_tensor_data_raw_average_per_channel ((gpointer) inptr, data_size,
        in_info->type, in_info->dimension, &average);
    /* calculate std only for default mode */
    if (filter->data_stand.mode == STAND_DEFAULT)
      gst_tensor_data_raw_std_per_channel ((gpointer) inptr, data_size,
          in_info->type, in_info->dimension, average, &std);
  } else {
    gst_tensor_data_raw_average ((gpointer) inptr, data_size,
        in_info->type, &average);
    /* calculate std only for default mode */
    if (filter->data_stand.mode == STAND_DEFAULT)
      gst_tensor_data_raw_std ((gpointer) inptr, data_size, in_info->type,
          average, &std);
  }

  switch (filter->data_stand.mode) {
    case STAND_DEFAULT:
    {
      if (!filter->data_stand.per_channel) {
        for (i = 0; i < num; i++) {
          data_idx = in_element_size * i;
          gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
              in_info->type, &tmp, _NNS_FLOAT64);

          tmp = fabs ((tmp - *average) / *std);

          data_idx = out_element_size * i;
          gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64,
              (gpointer) (outptr + data_idx), out_info->type);
        }
      } else {
        for (ch = 0; ch < ch_size; ++ch) {
          for (i = 0; i < num / ch_size; i++) {
            data_idx = in_element_size * ((i * ch_size) + ch);
            gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
                in_info->type, &tmp, _NNS_FLOAT64);

            tmp = fabs ((tmp - average[ch]) / std[ch]);

            data_idx = out_element_size * ((i * ch_size) + ch);
            gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64,
                (gpointer) (outptr + data_idx), out_info->type);
          }
        }
      }
      break;
    }
    case STAND_DC_AVERAGE:
    {
      if (!filter->data_stand.per_channel) {
        for (i = 0; i < num; i++) {
          data_idx = in_element_size * i;
          gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
              in_info->type, &tmp, _NNS_FLOAT64);

          tmp -= *average;

          data_idx = out_element_size * i;
          gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64,
              (gpointer) (outptr + data_idx), out_info->type);
        }
      } else {
        for (ch = 0; ch < ch_size; ++ch) {
          for (i = 0; i < num / ch_size; i++) {
            data_idx = in_element_size * ((i * ch_size) + ch);
            gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
                in_info->type, &tmp, _NNS_FLOAT64);

            tmp -= average[ch];

            data_idx = out_element_size * ((i * ch_size) + ch);
            gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64,
                (gpointer) (outptr + data_idx), out_info->type);
          }
        }
      }
      break;
    }
    default:
      GST_ERROR_OBJECT (filter, "Cannot identify mode\n");
      ret = GST_FLOW_ERROR;
  }

  g_free (average);
  g_free (std);

  return ret;
}

/**
 * @brief subrouting for tensor-tranform, "clamp" case.
 *        : pixel = if (pixel > max) ? max :
 *                  if (pixel < min) ? min : pixel
 * @param[in/out] filter "this" pointer
 * @param[in] in_info input tensor info
 * @param[in] out_info output tensor info
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_clamp (GstTensorTransform * filter,
    GstTensorInfo * in_info, GstTensorInfo * out_info,
    const uint8_t * inptr, uint8_t * outptr)
{
  gsize in_element_size, out_element_size;
  gulong i, num, data_idx;
  gdouble tmp;

  in_element_size = gst_tensor_get_element_size (in_info->type);
  out_element_size = gst_tensor_get_element_size (out_info->type);
  num = gst_tensor_get_element_count (in_info->dimension);

  for (i = 0; i < num; ++i) {
    data_idx = in_element_size * i;
    gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx), in_info->type,
        &tmp, _NNS_FLOAT64);

    tmp = CLAMP (tmp, filter->data_clamp.min, filter->data_clamp.max);

    data_idx = out_element_size * i;
    gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64, outptr + data_idx,
        out_info->type);
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
  GstTensorTransform *filter;
  GstTensorInfo *in_info, *out_info;
  GstFlowReturn res = GST_FLOW_ERROR;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMemory *out_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMapInfo in_map[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo out_map[NNS_TENSOR_SIZE_LIMIT];
  uint8_t *inptr, *outptr;
  guint i, num_tensors;
  gsize buf_size, hsize;
  GstTensorMetaInfo meta;
  GstTensorInfo in_flex_info, out_flex_info;
  gboolean in_flexible, out_flexible;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  g_return_val_if_fail (filter->loaded, GST_FLOW_ERROR);
  inbuf = gst_tensor_buffer_from_config (inbuf, &filter->in_config);

  in_flexible =
      gst_tensor_pad_caps_is_flexible (GST_BASE_TRANSFORM_SINK_PAD (trans));
  out_flexible =
      gst_tensor_pad_caps_is_flexible (GST_BASE_TRANSFORM_SRC_PAD (trans));

  if (in_flexible) {
    num_tensors = gst_buffer_n_memory (inbuf);
    g_return_val_if_fail (out_flexible, GST_FLOW_ERROR);
  } else {
    num_tensors = filter->in_config.info.num_tensors;
    g_return_val_if_fail (gst_buffer_n_memory (inbuf) == num_tensors,
        GST_FLOW_ERROR);
  }

  for (i = 0; i < num_tensors; i++) {
    in_info = &filter->in_config.info.info[i];
    out_info = &filter->out_config.info.info[i];

    if (filter->apply && !g_list_find (filter->apply, GINT_TO_POINTER (i))) {
      GstMemory *mem = gst_buffer_peek_memory (inbuf, i);

      if (!in_flexible && out_flexible) {
        /* append meta */
        gst_tensor_info_convert_to_meta (out_info, &meta);
        mem = gst_tensor_meta_info_append_header (&meta, mem);
      } else {
        mem = gst_memory_ref (mem);
      }

      gst_buffer_append_memory (outbuf, mem);
      continue;
    }

    /* parse input buffer */
    in_mem[i] = gst_buffer_peek_memory (inbuf, i);
    if (!gst_memory_map (in_mem[i], &in_map[i], GST_MAP_READ)) {
      ml_loge ("Cannot map input buffer to gst-buf at tensor-transform.\n");
      res = GST_FLOW_ERROR;
      goto done;
    }
    inptr = in_map[i].data;

    if (in_flexible) {
      in_info = &in_flex_info;
      out_info = &out_flex_info;

      gst_tensor_meta_info_parse_header (&meta, inptr);
      /** @todo max rank supported in tensor-transform is 4 */
      if (!gst_tensor_meta_info_convert (&meta, in_info)) {
        res = GST_FLOW_ERROR;
        goto done;
      }

      gst_tensor_transform_convert_dimension (filter, GST_PAD_SINK,
          i, in_info, out_info);

      hsize = gst_tensor_meta_info_get_header_size (&meta);
      inptr += hsize;
    }

    /* prepare output buffer */
    buf_size = gst_tensor_info_get_size (out_info);
    if (out_flexible) {
      gst_tensor_info_convert_to_meta (out_info, &meta);
      hsize = gst_tensor_meta_info_get_header_size (&meta);
      buf_size += hsize;
    }

    out_mem[i] = gst_allocator_alloc (NULL, buf_size, NULL);
    gst_buffer_append_memory (outbuf, out_mem[i]);

    if (!gst_memory_map (out_mem[i], &out_map[i], GST_MAP_WRITE)) {
      ml_loge ("Cannot map output buffer to gst-buf at tensor-transform.\n");
      res = GST_FLOW_ERROR;
      goto done;
    }
    outptr = out_map[i].data;

    if (out_flexible) {
      gst_tensor_meta_info_update_header (&meta, outptr);
      outptr += hsize;
    }

    switch (filter->mode) {
      case GTT_DIMCHG:
        res = gst_tensor_transform_dimchg (filter, in_info, out_info,
            inptr, outptr);
        break;
      case GTT_TYPECAST:
        res = gst_tensor_transform_typecast (filter, in_info, out_info,
            inptr, outptr);
        break;
      case GTT_ARITHMETIC:
        res = gst_tensor_transform_arithmetic (filter, in_info, out_info,
            inptr, outptr);
        break;
      case GTT_TRANSPOSE:
        res = gst_tensor_transform_transpose (filter, in_info, out_info,
            inptr, outptr);
        break;
      case GTT_STAND:
        res = gst_tensor_transform_stand (filter, in_info, out_info,
            inptr, outptr);
        break;
      case GTT_CLAMP:
        res = gst_tensor_transform_clamp (filter, in_info, out_info,
            inptr, outptr);
        break;
      default:
        ml_loge ("Not supported tensor transform mode");
        res = GST_FLOW_NOT_SUPPORTED;
        goto done;
    }
  }

done:
  for (i = 0; i < num_tensors; i++) {
    if (in_mem[i])
      gst_memory_unmap (in_mem[i], &in_map[i]);
    if (out_mem[i])
      gst_memory_unmap (out_mem[i], &out_map[i]);
  }

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
    const GstCaps * caps, GstTensorsConfig * config)
{
  GstStructure *structure;
  g_return_val_if_fail (config != NULL, FALSE);

  structure = gst_caps_get_structure (caps, 0);

  if (!gst_tensors_config_from_structure (config, structure)) {
    GST_WARNING_OBJECT (filter, "caps is not tensor %s\n",
        gst_structure_get_name (structure));
    return FALSE;
  }

  return gst_tensors_config_validate (config);
}

/**
 * @brief Dimension conversion calculation
 * @param[in] filter "this" pointer
 * @param[in] direction GST_PAD_SINK if input->output conv
 * @param[in] idx index of the input tensors
 * @param[in] in_info tensor info structure of source tensor (input if direction is SINK)
 * @param[out] out_info tensor info structure of destination tensor (output if direction is SINK)
 * @return TRUE if success
 */
static gboolean
gst_tensor_transform_convert_dimension (GstTensorTransform * filter,
    GstPadDirection direction, guint idx, const GstTensorInfo * in_info,
    GstTensorInfo * out_info)
{
  guint i;

  /* copy input info first, then update output info */
  gst_tensor_info_copy (out_info, in_info);

  if (filter->apply && !g_list_find (filter->apply, GINT_TO_POINTER (idx)))
    return TRUE;

  switch (filter->mode) {
    case GTT_DIMCHG:
    {
      unsigned int from = filter->data_dimchg.from;
      unsigned int to = filter->data_dimchg.to;

      if (direction == GST_PAD_SINK) {
        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          if ((i < from && i < to) || (i > from && i > to) || from == to) {
            out_info->dimension[i] = in_info->dimension[i];
          } else if (i == to) {
            out_info->dimension[i] = in_info->dimension[from];
          } else if (from > to) {
            g_assert (i > 0 && i > to);
            out_info->dimension[i] = in_info->dimension[i - 1];
          } else {
            g_assert (i < to && i < (NNS_TENSOR_RANK_LIMIT - 1));
            out_info->dimension[i] = in_info->dimension[i + 1];
          }
        }
      } else {
        for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
          if ((i < from && i < to) || (i > from && i > to) || from == to) {
            out_info->dimension[i] = in_info->dimension[i];
          } else if (i == from) {
            out_info->dimension[i] = in_info->dimension[to];
          } else if (from > to) {
            g_assert (i < from && i < (NNS_TENSOR_RANK_LIMIT - 1));
            out_info->dimension[i] = in_info->dimension[i + 1];
          } else {
            g_assert (i > 0 && i > from);
            out_info->dimension[i] = in_info->dimension[i - 1];
          }
        }
      }
      break;
    }
    case GTT_TYPECAST:
      /** For both directions, dimension does not change */
      if (direction == GST_PAD_SINK) {
        /** src = SINKPAD / dest = SRCPAD */
        out_info->type = filter->data_typecast.to;
      } else {
        /* cannot get the incoming data type on sink pad */
        out_info->type = _NNS_END;
      }
      break;

    case GTT_ARITHMETIC:
      /* check arith mode option has typecast operator */
      if (filter->data_arithmetic.out_type != _NNS_END) {
        if (direction == GST_PAD_SINK) {
          out_info->type = filter->data_arithmetic.out_type;
        } else {
          /* cannot get the incoming data type on sink pad */
          out_info->type = _NNS_END;
        }
      }
      break;

    case GTT_TRANSPOSE:
      if (direction == GST_PAD_SINK) {
        for (i = 0; i < NNS_TENSOR_TRANSPOSE_RANK_LIMIT; i++) {
          out_info->dimension[i] =
              in_info->dimension[filter->data_transpose.trans_order[i]];
        }
      } else {
        for (i = 0; i < NNS_TENSOR_TRANSPOSE_RANK_LIMIT; i++) {
          g_assert (filter->data_transpose.trans_order[i] <
              NNS_TENSOR_RANK_LIMIT);
          out_info->dimension[filter->data_transpose.trans_order[i]] =
              in_info->dimension[i];
        }
      }
      break;

    case GTT_STAND:
      /** For both directions, dimension does not change */
      if (direction == GST_PAD_SINK) {
        if (filter->data_stand.out_type != _NNS_END)
          out_info->type = filter->data_stand.out_type;
      } else {
        /* cannot get the incoming data type on sink pad */
        out_info->type = _NNS_END;
      }
      break;

    case GTT_CLAMP:
      /* same tensors info, do nothing. */
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
  GstCaps *result = NULL;
  GstStructure *structure;
  guint i, j;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug (filter, "Calling TransformCaps, direction = %d\n", direction);
  silent_debug_caps (filter, caps, "from");
  silent_debug_caps (filter, filtercap, "filter");

  result = gst_caps_new_empty ();
  for (i = 0; i < gst_caps_get_size (caps); i++) {
    GstTensorsConfig in_config, out_config;
    gboolean is_types_not_fixed = FALSE;
    GstCaps *result_aux = gst_caps_new_empty ();

    structure = gst_caps_get_structure (caps, i);

    gst_tensors_config_init (&in_config);
    gst_tensors_config_init (&out_config);

    gst_tensors_config_from_structure (&in_config, structure);

    if (gst_tensors_config_is_flexible (&in_config)) {
      /* output caps is also flexible */
      out_config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
    } else {
      for (j = 0; j < in_config.info.num_tensors; j++) {
        gst_tensor_transform_convert_dimension (filter, direction,
            j, &in_config.info.info[j], &out_config.info.info[j]);
        if (out_config.info.info[j].type == _NNS_END) {
          /* types cannot be specified */
          is_types_not_fixed = TRUE;
        }
      }
    }

    out_config.rate_d = in_config.rate_d;
    out_config.rate_n = in_config.rate_n;
    out_config.info.num_tensors = in_config.info.num_tensors;

    if (gst_structure_has_name (structure, NNS_MIMETYPE_TENSOR)) {
      gst_caps_append (result_aux, gst_tensor_caps_from_config (&out_config));
    } else {
      gst_caps_append (result_aux, gst_tensors_caps_from_config (&out_config));
    }

    /* remove `types` field from caps */
    if (is_types_not_fixed) {
      GstStructure *s;
      for (j = 0; j < gst_caps_get_size (result_aux); ++j) {
        s = gst_caps_get_structure (result_aux, j);
        gst_structure_remove_field (s, "types");
      }
    }

    gst_caps_append (result, result_aux);
  }

  if (filtercap && gst_caps_get_size (filtercap) > 0) {
    GstCaps *intersection;
    GstPad *pad;
    GstCaps *peer_caps;

    gst_tensor_caps_update_dimension (result, filtercap);

    intersection =
        gst_caps_intersect_full (result, filtercap, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;

    if (direction == GST_PAD_SINK)
      pad = GST_BASE_TRANSFORM_SRC_PAD (filter);
    else
      pad = GST_BASE_TRANSFORM_SINK_PAD (filter);

    if ((peer_caps = gst_pad_peer_query_caps (pad, NULL))) {
      gst_tensor_caps_update_dimension (result, peer_caps);
      gst_caps_unref (peer_caps);
    }
  }

  silent_debug_caps (filter, result, "to");
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

  silent_debug (filter, "Calling FixateCaps, direction = %d\n", direction);
  silent_debug_caps (filter, caps, "caps");
  silent_debug_caps (filter, othercaps, "othercaps");

  result =
      gst_tensor_transform_transform_caps (trans, direction, caps, othercaps);
  gst_caps_unref (othercaps);

  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  silent_debug_caps (filter, result, "result");
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
  GstTensorsConfig in_config, out_config;
  GstTensorsConfig config;
  gboolean in_flexible, out_flexible;
  gboolean allowed = FALSE;
  guint i;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug (filter, "Calling SetCaps\n");
  silent_debug_caps (filter, incaps, "incaps");
  silent_debug_caps (filter, outcaps, "outcaps");

  if (!gst_tensor_transform_read_caps (filter, incaps, &in_config)) {
    GST_ERROR_OBJECT (filter, "Cannot read cap of incaps\n");
    goto error;
  }

  if (!gst_tensor_transform_read_caps (filter, outcaps, &out_config)) {
    GST_ERROR_OBJECT (filter, "Cannot read cap of outcaps\n");
    goto error;
  }

  in_flexible = gst_tensors_config_is_flexible (&in_config);
  out_flexible = gst_tensors_config_is_flexible (&out_config);

  /* compare type and dimension */
  gst_tensors_config_init (&config);
  config.info.format = out_config.info.format;

  config.rate_n = in_config.rate_n;
  config.rate_d = in_config.rate_d;
  config.info.num_tensors = in_config.info.num_tensors;

  if (!in_flexible) {
    for (i = 0; i < in_config.info.num_tensors; i++) {
      if (!gst_tensor_transform_convert_dimension (filter, GST_PAD_SINK,
              i, &in_config.info.info[i], &config.info.info[i])) {
        GST_ERROR_OBJECT (filter,
            "Tensor info is not matched with given properties.");
        goto error;
      }
    }
  }

  if (out_flexible) {
    GST_INFO_OBJECT (filter, "Output tensor is flexible.");

    /* set output configuration if input is static */
    if (!in_flexible)
      out_config = config;
  } else if (!gst_tensors_config_is_equal (&out_config, &config)) {
    GST_ERROR_OBJECT (filter,
        "Tensor info is not matched with given properties.\n");
    goto error;
  }

  /* set in/out tensor info */
  filter->in_config = in_config;
  filter->out_config = out_config;
  allowed = TRUE;

error:
  if (!allowed)
    GST_ERROR_OBJECT (filter, "Set Caps Failed!\n");

  return allowed;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. Note that this is always the same with the input. optional vmethod of BaseTransform
 */
static gboolean
gst_tensor_transform_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize)
{
  UNUSED (trans);
  UNUSED (direction);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);
  /**
   * Consider multi-tensors.
   * Set each memory block in transform()
   */
  *othersize = 0;

  return TRUE;
}
