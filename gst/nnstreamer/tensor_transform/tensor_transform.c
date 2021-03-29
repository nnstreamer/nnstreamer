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
 * @file	tensor_transform.c
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
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT
#define REGEX_DIMCHG_OPTION "^([0-3]):([0-3])$"
#define REGEX_TYPECAST_OPTION "(^[u]?int(8|16|32|64)$|^float(32|64)$)"
#define REGEX_TRANSPOSE_OPTION "^(?:([0-2]):(?!.*\\1)){3}3$"
#define REGEX_CLAMP_OPTION "^((([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?))):"\
    "((([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)))$"
#define REGEX_ARITH_OPTION "^(typecast:([u]?int(8|16|32|64)|float(32|64)),)?"\
    "(((add|mul|div)(:([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?))+)(,|))+$"
#define REGEX_ARITH_OPTION_TYPECAST "(typecast:([u]?int(8|16|32|64)|float(32|64)))"

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
  PROP_APPLY
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
            "option=[typecast:TYPE,]add|mul|div:NUMBER..., ...",
          "arithmetic"},
      {GTT_TRANSPOSE, "Mode for transposing shape of tensor, "
            "option=D1\':D2\':D3\':D4 (fixed to 3)",
          "transpose"},
      {GTT_STAND, "Mode for statistical standardization of tensor, "
            "option=(default|dc-average)[:TYPE]",
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

#ifdef HAVE_ORC
/* define macros for orc */
/** @todo support 64bit integer and remove below line */
#define type_64bit_integer(t) ((t) == _NNS_INT64 || (t) == _NNS_UINT64)
#define orc_supported(f,itype,otype) ((f)->acceleration && !(type_64bit_integer (itype) || type_64bit_integer (otype)))

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
      default: GST_ERROR_OBJECT (filter, "Unsupported output type %d", otype); g_assert (0); break; \
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
      default: GST_ERROR_OBJECT (filter, "Unsupported input type %d", itype); g_assert (0); break; \
    } \
  } while (0)

#define orc_operator_func(i,n,v,opfunc) do { \
    switch ((v)->type) { \
      case _NNS_INT32: opfunc (s32) ((gpointer) i, (v)->data._int32_t, n); break; \
      case _NNS_UINT32: opfunc (u32) ((gpointer) i, (v)->data._uint32_t, n); break; \
      case _NNS_INT16: opfunc (s16) ((gpointer) i, (v)->data._int16_t, n); break; \
      case _NNS_UINT16: opfunc (u16) ((gpointer) i, (v)->data._uint16_t, n); break; \
      case _NNS_INT8: opfunc (s8) ((gpointer) i, (v)->data._int8_t, n); break; \
      case _NNS_UINT8: opfunc (u8) ((gpointer) i, (v)->data._uint8_t, n); break; \
      case _NNS_FLOAT64: opfunc (f64) ((gpointer) i, (v)->data._double, n); break; \
      case _NNS_FLOAT32: opfunc (f32) ((gpointer) i, (v)->data._float, n); break; \
      default: GST_ERROR_OBJECT (filter, "Unsupported type %d", (v)->type); g_assert (0); break; \
    } \
  } while (0)

#define orc_operator_div_loop(i,n,val,typename) do { \
    gsize idx; \
    typename *data_in = (typename *) (i); \
    for (idx = 0; idx < (n); ++idx) { \
      data_in[idx] = data_in[idx] / (val); \
    } \
  } while (0)

#define orc_operator(i,n,v,op) do { \
    switch (op) { \
      case GTT_OP_ADD: orc_operator_func (i, n, v, orc_func_add); break; \
      case GTT_OP_MUL: orc_operator_func (i, n, v, orc_func_mul); break; \
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

  if (strlen (filter->option) == 0) {
    ml_loge ("Given option string is empty, you should set proper option.");
    return FALSE;
  }

  filter_name = gst_object_get_name ((GstObject *) filter);

  switch (filter->mode) {
    case GTT_DIMCHG:
    {
      gchar **strv = NULL;

      if (!g_regex_match_simple (REGEX_DIMCHG_OPTION, filter->option, 0, 0)) {
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
      if (g_regex_match_simple (REGEX_TYPECAST_OPTION, filter->option, 0, 0)) {
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

      if (filter->operators) {
        GST_WARNING_OBJECT (filter,
            "There exists pre-defined operators (total %d), now reset these.",
            g_slist_length (filter->operators));

        g_slist_free_full (filter->operators, g_free);
        filter->operators = NULL;
      }

      regex_option_tc = g_regex_new (REGEX_ARITH_OPTION_TYPECAST, 0, 0, 0);

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

      if (!g_regex_match_simple (REGEX_ARITH_OPTION, str_option, 0, 0)) {
        ml_loge
            ("%s: arithmetic: \'%s\' is not valid option string: it should be in the form of [typecast:TYPE,]add|mul|div:NUMBER..., ...\n",
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
          op_s = g_new0 (tensor_transform_operator_s, 1);
          g_assert (op_s);

          op_s->op = gst_tensor_transform_get_operator (str_op[0]);

          switch (op_s->op) {
            case GTT_OP_TYPECAST:
              if (num_op > 1 && str_op[1]) {
                op_s->value.type = gst_tensor_get_type (str_op[1]);
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
                if (strchr (str_op[1], '.') || strchr (str_op[1], 'e') ||
                    strchr (str_op[1], 'E')) {
                  double val;

                  val = g_ascii_strtod (str_op[1], NULL);
                  gst_tensor_data_set (&op_s->value, _NNS_FLOAT64, &val);
                } else {
                  int64_t val;

                  val = g_ascii_strtoll (str_op[1], NULL, 10);
                  gst_tensor_data_set (&op_s->value, _NNS_INT64, &val);
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

      ret = filter->loaded = (filter->operators != NULL);
      g_strfreev (str_operators);
      g_free (str_option);
      break;
    }
    case GTT_TRANSPOSE:
    {
      int i;
      gchar **strv = NULL;

      if (!g_regex_match_simple (REGEX_TRANSPOSE_OPTION, filter->option, 0, 0)) {
        ml_loge
            ("%s: transpose: \'%s\' is not valid option string: it should be in the form of NEW_IDX_DIM0:NEW_IDX_DIM1:NEW_IDX_DIM2:3 (note that the index of the last dim is alwayes fixed to 3)\n",
            filter_name, filter->option);
        break;
      }

      strv = g_strsplit (filter->option, ":", NNS_TENSOR_RANK_LIMIT);
      for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
        filter->data_transpose.trans_order[i] =
            (uint8_t) g_ascii_strtoull (strv[i], NULL, 10);
      }

      ret = filter->loaded = TRUE;
      g_strfreev (strv);
      break;
    }
    case GTT_STAND:
    {
      gchar **strv = NULL;

      strv = g_strsplit (filter->option, ":", -1);

      filter->data_stand.mode = gst_tensor_transform_get_stand_mode (strv[0]);
      filter->data_stand.out_type = _NNS_END;
      if (g_strv_length (strv) > 1)
        filter->data_stand.out_type = gst_tensor_get_type (strv[1]);

      g_strfreev (strv);

      if (filter->data_stand.mode == STAND_END) {
        ml_loge
            ("%s: stand: \'%s\' is not valid option string: it should be in the form of (default|dc-average)[:TYPE]\n",
            filter_name, filter->option);
        break;
      }
      ret = filter->loaded = TRUE;
      break;
    }
    case GTT_CLAMP:
    {
      gchar **strv = NULL;

      if (!g_regex_match_simple (REGEX_CLAMP_OPTION, filter->option, 0, 0)) {
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
      if (TRUE == gst_tensor_transform_set_option_data (filter)) {
        silent_debug ("Option = %s --> %s\n", backup_option, filter->option);
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
      silent_debug ("acceleration = %d\n", filter->acceleration);
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
 * @param[in] idx index of the input tensors
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_dimchg (GstTensorTransform * filter,
    guint idx, const uint8_t * inptr, uint8_t * outptr)
{
  uint32_t *fromDim = filter->in_config.info.info[idx].dimension;
  uint32_t *toDim = filter->out_config.info.info[idx].dimension;
  tensor_type in_tensor_type = filter->in_config.info.info[idx].type;
  int from = filter->data_dimchg.from;
  int to = filter->data_dimchg.to;
  int i, j, k;
  unsigned int loopLimit = 1;
  gsize loopBlockSize = gst_tensor_get_element_size (in_tensor_type);
  gsize copyblocksize = gst_tensor_get_element_size (in_tensor_type);
  gsize copyblocklimit = 1;

  if (from == to) {
    /** Useless memcpy. Do not call this or @todo do "IP" operation */
    nns_memcpy (outptr, inptr,
        gst_tensor_info_get_size (&filter->in_config.info.info[idx]));
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
    ml_loge
        ("tensor-transform/dimchg operation is not permitted if from >= to.\n");
    return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * @brief subrouting for tensor-tranform, "typecast" case.
 * @param[in/out] filter "this" pointer
 * @param[in] idx index of the input tensors
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_typecast (GstTensorTransform * filter,
    guint idx, const uint8_t * inptr, uint8_t * outptr)
{
  gulong i, num;
  tensor_type in_tensor_type = filter->in_config.info.info[idx].type;
  tensor_type out_tensor_type = filter->out_config.info.info[idx].type;
  gsize in_element_size, out_element_size;

  num =
      gst_tensor_get_element_count (filter->in_config.info.info[idx].dimension);

#ifdef HAVE_ORC
  if (orc_supported (filter, in_tensor_type, out_tensor_type)) {
    orc_typecast (inptr, outptr, num, in_tensor_type, out_tensor_type);
    return GST_FLOW_OK;
  }
#endif

  in_element_size = gst_tensor_get_element_size (in_tensor_type);
  out_element_size = gst_tensor_get_element_size (out_tensor_type);

  for (i = 0; i < num; ++i) {
    gst_tensor_data_raw_typecast ((gpointer) (inptr + in_element_size * i),
        in_tensor_type, (gpointer) (outptr + out_element_size * i),
        out_tensor_type);
  }

  return GST_FLOW_OK;
}

/**
 * @brief subrouting for tensor-tranform, "arithmetic" case.
 * @param[in/out] filter "this" pointer
 * @param[in] idx index of the input tensors
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_arithmetic (GstTensorTransform * filter,
    guint idx, const uint8_t * inptr, uint8_t * outptr)
{
  gulong i, num;
  tensor_type in_tensor_type = filter->in_config.info.info[idx].type;
  tensor_type out_tensor_type = filter->out_config.info.info[idx].type;
  guint in_element_size, out_element_size;

  GSList *walk;
  tensor_transform_operator_s *op_s;
  tensor_data_s value;

  num =
      gst_tensor_get_element_count (filter->in_config.info.info[idx].dimension);

#ifdef HAVE_ORC
  if (orc_supported (filter, in_tensor_type, out_tensor_type)) {
    walk = filter->operators;

    /**
     * Typecast should be called at the first.
     * Do the typecast. If in/out type is same, this will copy the input array to output.
     */
    orc_typecast (inptr, outptr, num, in_tensor_type, out_tensor_type);

    while (walk) {
      op_s = (tensor_transform_operator_s *) walk->data;

      if (op_s->op != GTT_OP_TYPECAST) {
        gst_tensor_data_typecast (&op_s->value, out_tensor_type);
        orc_operator (outptr, num, &op_s->value, op_s->op);
      }

      walk = g_slist_next (walk);
    }

    return GST_FLOW_OK;
  }
#endif

  in_element_size = gst_tensor_get_element_size (in_tensor_type);
  out_element_size = gst_tensor_get_element_size (out_tensor_type);

  for (i = 0; i < num; ++i) {
    /* init value with input tensor type */
    gst_tensor_data_set (&value, in_tensor_type,
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
    g_assert (out_tensor_type == value.type);
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
            outidx = si*sj*sk*cl + sj*sk*ci + sk*cj+ck; \
            inidx = SK*SJ*SI*l + SJ*SI*k + SI*j + i; \
            _in = inptr + inidx * typesize; \
            _out = outptr + outidx *typesize; \
            nns_memcpy(_out, _in, typesize); \
	  }                                                      \
  } while(0);

/**
 * @brief subrouting for tensor-tranform, "transpose" case.
 * @param[in/out] filter "this" pointer
 * @param[in] idx index of the input tensors
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_transpose (GstTensorTransform * filter,
    guint idx, const uint8_t * inptr, uint8_t * outptr)
{
  int i, from, to;
  gboolean checkdim = FALSE;
  uint32_t *fromDim = filter->in_config.info.info[idx].dimension;
  tensor_type in_tensor_type = filter->in_config.info.info[idx].type;
  gsize type_size = gst_tensor_get_element_size (in_tensor_type);
  gsize indexI, indexJ, SL, SI, SJ, SK;
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
        gst_tensor_info_get_size (&filter->in_config.info.info[idx]));
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
 * @param[in] idx index of the input tensors
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_stand (GstTensorTransform * filter,
    guint idx, const uint8_t * inptr, uint8_t * outptr)
{
  tensor_type in_tensor_type = filter->in_config.info.info[idx].type;
  tensor_type out_tensor_type = filter->out_config.info.info[idx].type;
  gsize in_element_size, out_element_size, data_size;
  gulong i, num, data_idx;
  gdouble tmp, average, stand;

  in_element_size = gst_tensor_get_element_size (in_tensor_type);
  out_element_size = gst_tensor_get_element_size (out_tensor_type);
  num =
      gst_tensor_get_element_count (filter->in_config.info.info[idx].dimension);

  /* calc average */
  data_size = gst_tensor_info_get_size (&filter->in_config.info.info[idx]);
  average = gst_tensor_data_raw_average ((gpointer) inptr, data_size,
      in_tensor_type);

  switch (filter->data_stand.mode) {
    case STAND_DEFAULT:
    {
      stand = 0.0;

      for (i = 0; i < num; i++) {
        data_idx = in_element_size * i;
        gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
            in_tensor_type, &tmp, _NNS_FLOAT64);

        stand += pow (tmp - average, 2) / (num - 1);
      }

      stand = (stand != 0.0) ? sqrt (stand) : (1e-10);
      for (i = 0; i < num; i++) {
        data_idx = in_element_size * i;
        gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
            in_tensor_type, &tmp, _NNS_FLOAT64);

        tmp = fabs ((tmp - average) / stand);

        data_idx = out_element_size * i;
        gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64,
            (gpointer) (outptr + data_idx), out_tensor_type);
      }

      break;
    }
    case STAND_DC_AVERAGE:
    {
      for (i = 0; i < num; i++) {
        data_idx = in_element_size * i;
        gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx),
            in_tensor_type, &tmp, _NNS_FLOAT64);

        tmp -= average;

        data_idx = out_element_size * i;
        gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64,
            (gpointer) (outptr + data_idx), out_tensor_type);
      }
      break;
    }
    default:
      GST_ERROR_OBJECT (filter, "Cannot identify mode\n");
      return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * @brief subrouting for tensor-tranform, "clamp" case.
 *        : pixel = if (pixel > max) ? max :
 *                  if (pixel < min) ? min : pixel
 * @param[in/out] filter "this" pointer
 * @param[in] idx index of the input tensors
 * @param[in] inptr input tensor
 * @param[out] outptr output tensor
 * @return Gst flow status
 */
static GstFlowReturn
gst_tensor_transform_clamp (GstTensorTransform * filter,
    guint idx, const uint8_t * inptr, uint8_t * outptr)
{
  tensor_type in_tensor_type = filter->in_config.info.info[idx].type;
  tensor_type out_tensor_type = filter->out_config.info.info[idx].type;

  gsize in_element_size, out_element_size;
  gulong i, num, data_idx;
  gdouble tmp;

  in_element_size = gst_tensor_get_element_size (in_tensor_type);
  out_element_size = gst_tensor_get_element_size (out_tensor_type);
  num =
      gst_tensor_get_element_count (filter->in_config.info.info[idx].dimension);

  for (i = 0; i < num; ++i) {
    data_idx = in_element_size * i;
    gst_tensor_data_raw_typecast ((gpointer) (inptr + data_idx), in_tensor_type,
        &tmp, _NNS_FLOAT64);

    tmp = CLAMP (tmp, filter->data_clamp.min, filter->data_clamp.max);

    data_idx = out_element_size * i;
    gst_tensor_data_raw_typecast (&tmp, _NNS_FLOAT64, outptr + data_idx,
        out_tensor_type);
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
  GstFlowReturn res = GST_FLOW_ERROR;
  GstTensorTransform *filter;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMemory *out_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo out_info[NNS_TENSOR_SIZE_LIMIT];
  uint8_t *inptr, *outptr;
  guint i, num_tensors;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  g_return_val_if_fail (filter->loaded, GST_FLOW_ERROR);

  num_tensors = filter->in_config.info.num_tensors;
  g_return_val_if_fail (gst_buffer_n_memory (inbuf) == num_tensors,
      GST_FLOW_ERROR);

  for (i = 0; i < num_tensors; i++) {
    gsize buf_size = 0;

    if (filter->apply && !g_list_find (filter->apply, GINT_TO_POINTER (i))) {
      gst_buffer_append_memory (outbuf, gst_buffer_get_memory (inbuf, i));
      continue;
    }

    in_mem[i] = gst_buffer_peek_memory (inbuf, i);
    if (FALSE == gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
      ml_loge ("Cannot map input buffer to gst-buf at tensor-transform.\n");
      res = GST_FLOW_ERROR;
      goto done;
    }
    inptr = in_info[i].data;

    buf_size = gst_tensor_info_get_size (&filter->out_config.info.info[i]);
    out_mem[i] = gst_allocator_alloc (NULL, buf_size, NULL);
    gst_buffer_append_memory (outbuf, out_mem[i]);

    if (FALSE == gst_memory_map (out_mem[i], &out_info[i], GST_MAP_READ)) {
      ml_loge ("Cannot map output buffer to gst-buf at tensor-transform.\n");
      res = GST_FLOW_ERROR;
      goto done;
    }
    outptr = out_info[i].data;

    switch (filter->mode) {
      case GTT_DIMCHG:
        res = gst_tensor_transform_dimchg (filter, i, inptr, outptr);
        break;
      case GTT_TYPECAST:
        res = gst_tensor_transform_typecast (filter, i, inptr, outptr);
        break;
      case GTT_ARITHMETIC:
        res = gst_tensor_transform_arithmetic (filter, i, inptr, outptr);
        break;
      case GTT_TRANSPOSE:
        res = gst_tensor_transform_transpose (filter, i, inptr, outptr);
        break;
      case GTT_STAND:
        res = gst_tensor_transform_stand (filter, i, inptr, outptr);
        break;
      case GTT_CLAMP:
        res = gst_tensor_transform_clamp (filter, i, inptr, outptr);
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
      gst_memory_unmap (in_mem[i], &in_info[i]);
    if (out_mem[i])
      gst_memory_unmap (out_mem[i], &out_info[i]);
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

  return gst_tensors_info_validate (&config->info);
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
      int from = filter->data_dimchg.from;
      int to = filter->data_dimchg.to;

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
  const gchar *caps_name;
  guint i, j;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug ("Calling TransformCaps, direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filtercap, "filter");

  result = gst_caps_new_empty ();
  for (i = 0; i < gst_caps_get_size (caps); i++) {
    GstTensorsConfig in_config, out_config;

    structure = gst_caps_get_structure (caps, i);
    caps_name = gst_structure_get_name (structure);

    gst_tensors_config_init (&in_config);
    gst_tensors_config_init (&out_config);

    gst_tensors_config_from_structure (&in_config, structure);

    for (j = 0; j < in_config.info.num_tensors; j++) {
      gst_tensor_transform_convert_dimension (filter, direction,
          j, &in_config.info.info[j], &out_config.info.info[j]);
    }
    out_config.rate_d = in_config.rate_d;
    out_config.rate_n = in_config.rate_n;
    out_config.info.num_tensors = in_config.info.num_tensors;

    if (g_strcmp0 (caps_name, "other/tensor") == 0) {
      GstTensorConfig tensor_config;

      gst_tensor_config_init (&tensor_config);
      tensor_config.info = out_config.info.info[0];
      tensor_config.rate_n = out_config.rate_n;
      tensor_config.rate_d = out_config.rate_d;

      gst_caps_append (result, gst_tensor_caps_from_config (&tensor_config));
    } else {
      gst_caps_append (result, gst_tensors_caps_from_config (&out_config));
    }
  }

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
  GstTensorsConfig in_config, out_config;
  GstTensorsConfig config;
  gboolean allowed = FALSE;
  guint i;

  filter = GST_TENSOR_TRANSFORM_CAST (trans);

  silent_debug ("Calling SetCaps\n");
  silent_debug_caps (incaps, "incaps");
  silent_debug_caps (outcaps, "outcaps");

  if (!gst_tensor_transform_read_caps (filter, incaps, &in_config) ||
      !gst_tensors_config_validate (&in_config)) {
    GST_ERROR_OBJECT (filter, "Cannot read cap of incaps\n");
    goto error;
  }

  if (!gst_tensor_transform_read_caps (filter, outcaps, &out_config) ||
      !gst_tensors_config_validate (&out_config)) {
    GST_ERROR_OBJECT (filter, "Cannot read cap of outcaps\n");
    goto error;
  }

  /* compare type and dimension */
  for (i = 0; i < in_config.info.num_tensors; i++) {
    if (!gst_tensor_transform_convert_dimension (filter, GST_PAD_SINK,
            i, &in_config.info.info[i], &config.info.info[i])) {
      GST_ERROR_OBJECT (filter,
          "Tensor info is not matched with given properties.");
      goto error;
    }
  }

  config.rate_n = in_config.rate_n;
  config.rate_d = in_config.rate_d;
  config.info.num_tensors = in_config.info.num_tensors;

  if (!gst_tensors_config_is_equal (&out_config, &config)) {
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
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  /**
   * Consider multi-tensors.
   * Set each memory block in transform()
   */
  *othersize = 0;

  return TRUE;
}
