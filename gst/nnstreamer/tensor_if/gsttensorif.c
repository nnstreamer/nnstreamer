/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-IF
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	gsttensorif.c
 * @date	08 April 2020
 * @brief	GStreamer plugin to control flow based on tensor values
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_if
 *
 * A filter that controls its src-pad based on the values (other/tensor(s))
 * of its sink-pad.
 * For example, you may skip frames if there is no object detected with
 * high confidence.
 *
 * The format of statement with tensor-if is:
 * if (Compared_Value OPERATOR Supplied_Value(s))) then THEN else ELSE
 * Compared_Value and Supplied_Value are the operands.
 * Compared_Value is a value from input tensor(s).
 * SUpplied_Value is a value from tensor-if properties.
 *
 * If the given if-condition is simple enough (e.g., if a specific element
 * is between a given range in a tensor frame), it can be expressed as:
 * <refsect2>
 * <title>Example launch line with simple if condition</title>
 * gst-launch ... (some tensor stream) !
 *      tensor_if name=tif
 *        compared_value=A_VALUE compared_value_option=3:4:2:5,0
 *        operator=RANGE_INCLUSIVE
 *        supplied_values=10,100
 *        then=PASSTHROUGH
 *        else=TENSORPICK
 *        else_option=1
 *      tif.src_0 ! queue ! (tensor(s) stream for TRUE action) ...
 *      tif.src_1 ! queue ! (tensor(s) stream for FALSE action) ...
 * </refsect2>
 *
 * However, if the if-condition is complex and cannot be expressed with
 * tensor-if expressions, you may create a corresponding custom filter
 * with tensor-filter, whose output is other/tensors with an additional tensor
 * that is "1:1:1:1, uint8", which is 1 (true) or 0 (false) as the
 * first tensor of other/tensors and the input tensor/tensors.
 * Then, you can create a pipeline as follows:
 * <refsect2>
 * <title>Example launch line with complex if condition</title>
 * gst-launch ... (some tensor stream)
 *      ! tensor_filter framework=custom name=your_code.so
 *      ! tensor_if compared_value=A_VALUE
 *          compared_value_option=0:0:0:0,0 # 1st tensor's [0][0][0][0].
 *          operator=EQ
 *          supplied_values=1
 *          then=PASSTHROUGH # or whatsoever you want
 *          else=SKIP # or whatsoever you want
 *      ! tensor_demux name=d
 *        d.src_0 ! queue ! fakesink # throw away the 1/0 value.
 *        d.src_1 ! queue ! do whatever you want here...
 *        ...
 * </refsect2>
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <nnstreamer_log.h>
#include <string.h>

#include "gsttensorif.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!tensor_if->silent)
#endif

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
        GST_DEBUG_OBJECT (tensor_if, msg " = %s\n", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

/**
 * @brief tensor_if properties
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_CV, /**< Compared Value, operand 1 (from input tensor(s)) */
  PROP_CV_OPTION, /**< Compared Value Option */
  PROP_OP, /**< Operator */
  PROP_SV, /**< Supplied Value, operand 2 (from the properties) */
  PROP_THEN, /**< Action if it is TRUE */
  PROP_THEN_OPTION, /**< Option for TRUE Action */
  PROP_ELSE, /**< Action if it is FALSE */
  PROP_ELSE_OPTION, /**< Option for FALSE Action */
};

GST_DEBUG_CATEGORY_STATIC (gst_tensor_if_debug);
#define GST_CAT_DEFAULT gst_tensor_if_debug

#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT
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
    GST_PAD_SOMETIMES,
    GST_STATIC_CAPS (CAPS_STRING));

#define gst_tensor_if_parent_class parent_class
G_DEFINE_TYPE (GstTensorIf, gst_tensor_if, GST_TYPE_ELEMENT);

/* GObject vmethod implementations */
static void gst_tensor_if_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_if_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_tensor_if_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);
static gboolean gst_tensor_if_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static void gst_tensor_if_dispose (GObject * object);

static void gst_tensor_if_install_properties (GObjectClass * gobject_class);

#define GST_TYPE_TENSOR_IF_CV (gst_tensor_if_cv_get_type ())
/**
 * @brief A private function to register GEnumValue array for the 'compared_value' property
 *        to a GType and return it
 */
static GType
gst_tensor_if_cv_get_type (void)
{
  static GType mode_type = 0;

  if (mode_type == 0) {
    static GEnumValue mode_types[] = {
      {TIFCV_A_VALUE, "A_VALUE", "Decide based on a single scalar value"},
      {TIFCV_TENSOR_AVERAGE_VALUE, "TENSOR_AVERAGE_VALUE",
          "Decide based on a average value of a specific tensor"},
      {0, NULL, NULL},
    };
    mode_type = g_enum_register_static ("tensor_if_compared_value", mode_types);
  }

  return mode_type;
}

#define GST_TYPE_TENSOR_IF_OP (gst_tensor_if_op_get_type ())
/**
 * @brief A private function to register GEnumValue array for the 'operator' property
 *        to a GType and return it
 */
static GType
gst_tensor_if_op_get_type (void)
{
  static GType mode_type = 0;

  if (mode_type == 0) {
    static GEnumValue mode_types[] = {
      {TIFOP_EQ, "EQ", "eqaual"},
      {TIFOP_NE, "NE", "not_eqaual"},
      {TIFOP_GT, "GT", "greater_than"},
      {TIFOP_GE, "GE", "greater_or_equal"},
      {TIFOP_LT, "LT", "less_than"},
      {TIFOP_LE, "LE", "less_or_equal"},
      {TIFOP_RANGE_INCLUSIVE, "RANGE_INCLUSIVE", "range inclusive"},
      {TIFOP_RANGE_EXCLUSIVE, "RANGE_EXCLUSIVE", "range exclusive"},
      {TIFOP_NOT_IN_RANGE_INCLUSIVE, "NOT_IN_RANGE_INCLUSIVE",
          "not in range inclusive"},
      {TIFOP_NOT_IN_RANGE_EXCLUSIVE, "NOT_IN_RANGE_EXCLUSIVE",
          "not in range exclusive"},
      {0, NULL, NULL},
    };
    mode_type = g_enum_register_static ("tensor_if_operator", mode_types);
  }

  return mode_type;
}

#define GST_TYPE_TENSOR_IF_ACT (gst_tensor_if_act_get_type ())
/**
 * @brief A private function to register GEnumValue array for the 'then' and 'else' properties
 *        to a GType and return it
 */
static GType
gst_tensor_if_act_get_type (void)
{
  static GType mode_type = 0;

  if (mode_type == 0) {
    static GEnumValue mode_types[] = {
      {TIFB_PASSTHROUGH, "PASSTHROUGH", "passthrough"},
      {TIFB_SKIP, "SKIP", "skip"},
      {TIFB_TENSORPICK, "TENSORPICK", "tensorpick"},
      {0, NULL, NULL},
    };
    mode_type = g_enum_register_static ("tensor_if_behavior", mode_types);
  }

  return mode_type;
}

/**
 * @brief initialize the tensor_if's class (GST Standard)
 */
static void
gst_tensor_if_class_init (GstTensorIfClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_if_debug, "tensor_if", 0,
      "Tensor if to control streams based on tensor(s) values");

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->set_property = gst_tensor_if_set_property;
  gobject_class->get_property = gst_tensor_if_get_property;
  gobject_class->dispose = gst_tensor_if_dispose;

  gst_tensor_if_install_properties (gobject_class);

  gst_element_class_set_details_simple (gstelement_class,
      "Tensor_If",
      "NNStreamer/If",
      "Controls streams based on the tensor(s) values",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
}

/**
 * @brief initialize the new element (GST Standard)
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_if_init (GstTensorIf * tensor_if)
{
  tensor_if->silent = TRUE;
  gst_tensors_config_init (&tensor_if->in_config);
  gst_tensors_config_init (&tensor_if->out_config[0]);
  gst_tensors_config_init (&tensor_if->out_config[1]);

  tensor_if->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_if), tensor_if->sinkpad);
  gst_pad_set_chain_function (tensor_if->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_if_chain));
  gst_pad_set_event_function (tensor_if->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_if_event));

  tensor_if->num_srcpads = 0;
  tensor_if->srcpads = NULL;
}

/**
 * @brief dispose function for tensor if (gst element vmethod)
 */
static void
gst_tensor_if_dispose (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->dispose (object);
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
gst_tensor_if_typecast_value (GstTensorIf * tensor_if,
    tensor_if_data_s * value, tensor_type type)
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
          typecast_value (value, int64_t);
          value->type = _NNS_INT64;
        }
        typecast_value (value, uint64_t);
        break;
      default:
        GST_ERROR_OBJECT (tensor_if, "Unknown tensor type %d", type);
        return FALSE;
    }

    value->type = type;
  }

  return TRUE;
}

/**
 * @brief Macro to set tensor_if data
 */
#define set_tensor_data(v,d,vtype) do { \
    (v)->data._##vtype = *((vtype *) d); \
  } while (0)

/**
 * @brief Set tensor element data with given type
 * @param value struct for tesnor_if data
 * @param data pointer of tensor element value
 * @return TRUE if no error
 */
static gboolean
gst_tensor_if_set_data (tensor_if_data_s * value, gpointer data)
{
  g_return_val_if_fail (value != NULL, FALSE);
  g_return_val_if_fail (data != NULL, FALSE);

  switch (value->type) {
    case _NNS_INT32:
      set_tensor_data (value, data, int32_t);
      break;
    case _NNS_UINT32:
      set_tensor_data (value, data, uint32_t);
      break;
    case _NNS_INT16:
      set_tensor_data (value, data, int16_t);
      break;
    case _NNS_UINT16:
      set_tensor_data (value, data, uint16_t);
      break;
    case _NNS_INT8:
      set_tensor_data (value, data, int8_t);
      break;
    case _NNS_UINT8:
      set_tensor_data (value, data, uint8_t);
      break;
    case _NNS_FLOAT64:
      set_tensor_data (value, data, double);
      break;
    case _NNS_FLOAT32:
      set_tensor_data (value, data, float);
      break;
    case _NNS_INT64:
      set_tensor_data (value, data, int64_t);
      break;
    case _NNS_UINT64:
      set_tensor_data (value, data, uint64_t);
      break;
    default:
      return FALSE;
  }
  return TRUE;
}

/**
 * @brief Convert GValue to GList according to delimiters
 */
static void
gst_tensor_if_set_property_glist (const GValue * value, GList ** prop_list,
    const gchar * delimiters)
{
  gint64 val;
  const gchar *param = g_value_get_string (value);
  gchar **strv = g_strsplit_set (param, delimiters, -1);
  gint i, num = g_strv_length (strv);
  *prop_list = NULL;

  for (i = 0; i < num; i++) {
    val = g_ascii_strtoll (strv[i], NULL, 10);
    if (errno == ERANGE) {
      ml_loge ("Overflow occured during converting %s to a gint64 value",
          strv[i]);
    }
    *prop_list = g_list_append (*prop_list, GINT_TO_POINTER (val));
  }
  g_strfreev (strv);
}

/**
 * @brief Convert GValue to GList according to delimiters
 */
static void
gst_tensor_if_set_property_supplied_value (const GValue * value,
    tensor_if_sv_s * sv, const gchar * delimiters)
{
  gint i;
  gboolean is_float = FALSE;
  const gchar *param = g_value_get_string (value);
  gchar **strv = g_strsplit_set (param, delimiters, -1);
  gint num = g_strv_length (strv);

  if (strchr (param, '.') || strchr (param, 'E') || strchr (param, 'e')) {
    is_float = TRUE;
  }

  sv->num = num;
  for (i = 0; i < num; i++) {
    if (is_float == TRUE) {
      sv->type = _NNS_FLOAT64;
      sv->data[i]._double = g_ascii_strtod (strv[i], NULL);
    } else {
      sv->type = _NNS_INT64;
      sv->data[i]._int64_t = g_ascii_strtoll (strv[i], NULL, 10);
    }
  }
  g_strfreev (strv);
}

/**
 * @brief Setter for tensor_if properties.
 */
static void
gst_tensor_if_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorIf *self = GST_TENSOR_IF (object);

  switch (prop_id) {
    case PROP_CV:
      self->cv = g_value_get_enum (value);
      break;
    case PROP_CV_OPTION:
      gst_tensor_if_set_property_glist (value, &self->cv_option, ":,");
      break;
    case PROP_OP:
      self->op = g_value_get_enum (value);
      break;
    case PROP_SV:
      gst_tensor_if_set_property_supplied_value (value, self->sv, ",");
      break;
    case PROP_THEN:
      self->act_then = g_value_get_enum (value);
      break;
    case PROP_THEN_OPTION:
      gst_tensor_if_set_property_glist (value, &self->then_option, ",");
      break;
    case PROP_ELSE:
      self->act_else = g_value_get_enum (value);
      break;
    case PROP_ELSE_OPTION:
      gst_tensor_if_set_property_glist (value, &self->else_option, ",");
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Convert GList to GValue
 */
static void
gst_tensor_if_property_to_string (GValue * value, GList * prop_list,
    guint prop_id)
{
  GList *list;
  gchar *p;
  GPtrArray *arr = g_ptr_array_new ();
  gchar **strings;
  guint len;

  for (list = prop_list; list != NULL; list = list->next) {
    g_ptr_array_add (arr, g_strdup_printf ("%i", GPOINTER_TO_INT (list->data)));
  }
  g_ptr_array_add (arr, NULL);
  strings = (gchar **) g_ptr_array_free (arr, FALSE);
  len = g_strv_length (strings);
  if (prop_id == PROP_CV_OPTION && len % 5 == 0) {
    gchar *dim =
        g_strjoin (":", strings[0], strings[1], strings[2], strings[3], NULL);
    p = g_strjoin (",", dim, strings[4], NULL);
    g_free (dim);
  } else {
    p = g_strjoinv (",", strings);
  }

  g_strfreev (strings);
  g_value_take_string (value, p);
}

/**
 * @brief Convert GValue to supplied value according to delimiters
 */
static void
gst_tensor_if_get_property_supplied_value (GValue * value, tensor_if_sv_s * sv)
{
  gint i;
  gchar *p;
  GPtrArray *arr = g_ptr_array_new ();
  gchar **strings;

  for (i = 0; i < sv->num; i++) {
    if (sv->type == _NNS_FLOAT64) {
      g_ptr_array_add (arr, g_strdup_printf ("%lf", sv->data[i]._double));
    } else {
      g_ptr_array_add (arr, g_strdup_printf ("%ld",
              (long int) sv->data[i]._int64_t));
    }
  }
  g_ptr_array_add (arr, NULL);
  strings = (gchar **) g_ptr_array_free (arr, FALSE);
  p = g_strjoinv (",", strings);
  g_strfreev (strings);
  g_value_take_string (value, p);
}

/**
 * @brief Getter for tensor_if properties.
 */
static void
gst_tensor_if_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorIf *self = GST_TENSOR_IF (object);

  switch (prop_id) {
    case PROP_CV:
      g_value_set_enum (value, self->cv);
      break;
    case PROP_CV_OPTION:
      gst_tensor_if_property_to_string (value, self->cv_option, prop_id);
      break;
    case PROP_OP:
      g_value_set_enum (value, self->op);
      break;
    case PROP_SV:
      gst_tensor_if_get_property_supplied_value (value, self->sv);
      break;
    case PROP_THEN:
      g_value_set_enum (value, self->act_then);
      break;
    case PROP_THEN_OPTION:
      gst_tensor_if_property_to_string (value, self->then_option, prop_id);
      break;
    case PROP_ELSE:
      g_value_set_enum (value, self->act_else);
      break;
    case PROP_ELSE_OPTION:
      gst_tensor_if_property_to_string (value, self->else_option, prop_id);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Installs all the properties for tensor_if
 * @param[in] gobject_class Glib object class whose properties will be set
 */
static void
gst_tensor_if_install_properties (GObjectClass * gobject_class)
{
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          FALSE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CV,
      g_param_spec_enum ("compared-value", "CV",
          "Compared value from input tensor(s)", GST_TYPE_TENSOR_IF_CV,
          TIFCV_A_VALUE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CV_OPTION,
      g_param_spec_string ("compared-value-option", "CV_OPTION",
          "Specify an element of the nth tensor or pick tensor ", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SV,
      g_param_spec_string ("supplied-value", "SV",
          " Supplied Value by user ", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OP,
      g_param_spec_enum ("operator", "OP", "Comparison Operator",
          GST_TYPE_TENSOR_IF_OP, TIFOP_EQ,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_THEN,
      g_param_spec_enum ("then", "THEN", "Action if it is TRUE",
          GST_TYPE_TENSOR_IF_ACT, TIFB_PASSTHROUGH,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_THEN_OPTION,
      g_param_spec_string ("then-option", "THEN_OPTION",
          "Pick tensor ", "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ELSE,
      g_param_spec_enum ("else", "ELSE", "Action if it is FALSE",
          GST_TYPE_TENSOR_IF_ACT, TIFB_SKIP,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ELSE_OPTION,
      g_param_spec_string ("else-option", "ELSE_OPTION",
          "Pick tensor ", "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

/**
 * @brief Parse caps and configure tensors info.
 * @param tensor_if GstTensorIf Ojbect
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully configured, return TRUE)
 */
static gboolean
gst_tensor_if_parse_caps (GstTensorIf * tensor_if, GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsConfig *config;

  config = &tensor_if->in_config;
  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (config, structure);

  return gst_tensors_config_validate (config);
}

/**
 * @brief event function for sink (gst element vmethod)
 */
static gboolean
gst_tensor_if_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorIf *tensor_if;
  tensor_if = GST_TENSOR_IF (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      if (!gst_tensor_if_parse_caps (tensor_if, caps)) {
        GST_ERROR_OBJECT (tensor_if, "Failed to parse caps.\n");
        return FALSE;
      }
      break;
    }
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief Check whether caps is other/tensor or not
 * @return TRUE if other/tensor, FALSE if not
 */
static gboolean
gst_tensor_if_is_tensor_caps (GstCaps * caps)
{
  GstStructure *caps_s;
  guint i, caps_len;

  caps_len = gst_caps_get_size (caps);

  for (i = 0; i < caps_len; i++) {
    caps_s = gst_caps_get_structure (caps, i);
    if (gst_structure_has_name (caps_s, "other/tensor")) {
      return TRUE;
    }
  }
  return FALSE;
}

/**
 * @brief Checking if the source pad is created and if not, create TensorPad
 * @param tesnor_if TensorIf Object
 * @param config Tensors Config Data
 * @param nth source ordering
 * @return TensorPad if pad is already created, then return created pad.
 *         If not return new pad after creation.
 */
static GstTensorPad *
gst_tensor_if_get_tensor_pad (GstTensorIf * tensor_if,
    GstTensorsConfig * config, gboolean * created, gint nth)
{
  GSList *walk;
  GstPad *pad;
  GstTensorPad *tensorpad;
  gchar *name;
  GstCaps *peer_caps, *caps = NULL;

  walk = tensor_if->srcpads;
  while (walk) {
    GstTensorPad *pad = (GstTensorPad *) walk->data;
    if (nth == pad->nth) {
      if (created) {
        *created = FALSE;
      }
      return pad;
    }
    walk = walk->next;
  }

  tensorpad = g_new0 (GstTensorPad, 1);
  g_assert (tensorpad != NULL);
  GST_DEBUG_OBJECT (tensor_if, "createing pad: %d(%dth)",
      tensor_if->num_srcpads, nth);

  name = g_strdup_printf ("src_%d", nth);
  pad = gst_pad_new_from_static_template (&src_factory, name);
  g_free (name);

  tensorpad->pad = pad;
  tensorpad->nth = nth;
  tensorpad->last_ret = GST_FLOW_OK;
  tensorpad->last_ts = GST_CLOCK_TIME_NONE;

  tensor_if->srcpads = g_slist_append (tensor_if->srcpads, tensorpad);
  tensor_if->num_srcpads++;

  gst_pad_use_fixed_caps (pad);
  gst_pad_set_active (pad, TRUE);
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_if), pad);

  peer_caps = gst_pad_peer_query_caps (pad, NULL);
  silent_debug_caps (peer_caps, "peer_caps");

  if (config->info.num_tensors == 1 && gst_tensor_if_is_tensor_caps (peer_caps)) {
    GstTensorConfig tensor_config;
    tensor_config.info = config->info.info[0];
    tensor_config.rate_n = config->rate_n;
    tensor_config.rate_d = config->rate_d;
    caps = gst_tensor_caps_from_config (&tensor_config);
  }

  if (caps == NULL) {
    caps = gst_tensors_caps_from_config (config);
  }
  silent_debug_caps (caps, "out caps");
  gst_pad_set_caps (pad, caps);

  gst_caps_unref (peer_caps);
  gst_caps_unref (caps);

  if (created) {
    *created = TRUE;
  }

  return tensorpad;
}

/**
 * @brief Check the status among sources in if
 * @param tensor_if TensorIf Object
 * @param TensorPad Tensorpad
 * @param ret return status of current pad
 * @return return status after check sources
 */
static GstFlowReturn
gst_tensor_if_combine_flows (GstTensorIf * tensor_if,
    GstTensorPad * pad, GstFlowReturn ret)
{
  GSList *walk;
  pad->last_ret = ret;

  if (ret != GST_FLOW_NOT_LINKED)
    goto done;

  for (walk = tensor_if->srcpads; walk; walk = g_slist_next (walk)) {
    GstTensorPad *opad = (GstTensorPad *) walk->data;
    ret = opad->last_ret;
    if (ret != GST_FLOW_NOT_LINKED)
      goto done;
  }
done:
  return ret;
}

/**
 * @brief Macro for operator function.
 */
#define operator_func(cv,t,op,sv1,sv2,ret) do { \
  switch (op) { \
    case TIFOP_EQ: ret = (cv._##t == sv1._##t) ? TRUE : FALSE; break; \
    case TIFOP_NE: ret = (cv._##t != sv1._##t) ? TRUE : FALSE; break; \
    case TIFOP_GT: ret = (cv._##t > sv1._##t) ? TRUE : FALSE; break; \
    case TIFOP_GE: ret = (cv._##t >= sv1._##t) ? TRUE : FALSE; break; \
    case TIFOP_LT: ret = (cv._##t < sv1._##t) ? TRUE : FALSE; break; \
    case TIFOP_LE: ret = (cv._##t <= sv1._##t) ? TRUE : FALSE; break; \
    case TIFOP_RANGE_INCLUSIVE: \
      ret = (sv1._##t <= cv._##t && cv._##t <= sv2._##t) ? TRUE : FALSE; break; \
    case TIFOP_RANGE_EXCLUSIVE: \
      ret = (sv1._##t < cv._##t && cv._##t < sv2._##t) ? TRUE : FALSE; break; \
    case TIFOP_NOT_IN_RANGE_INCLUSIVE: \
      ret = (cv._##t < sv1._##t && sv2._##t < cv._##t) ? TRUE : FALSE; break; \
    case TIFOP_NOT_IN_RANGE_EXCLUSIVE: \
      ret = (cv._##t <= sv1._##t && sv2._##t <= cv._##t) ? TRUE : FALSE; break; \
    default: break; \
  } \
} while (0)

/**
 * @brief Get comparison value
 */
static gboolean
gst_tensor_if_get_comparison_result (GstTensorIf * tensor_if,
    tensor_if_data_s * cv, gboolean * result)
{
  gboolean ret = FALSE;
  tensor_if_data_s svtc_1, svtc_2;

  svtc_1.type = tensor_if->sv->type;
  svtc_1.data = tensor_if->sv->data[0];
  gst_tensor_if_typecast_value (tensor_if, &svtc_1, cv->type);

  if (tensor_if->sv->num > 1) {
    svtc_2.type = tensor_if->sv->type;
    svtc_2.data = tensor_if->sv->data[1];
    gst_tensor_if_typecast_value (tensor_if, &svtc_2, cv->type);
  }

  switch (cv->type) {
    case _NNS_INT32:
      operator_func (cv->data, int32_t, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_UINT32:
      operator_func (cv->data, uint32_t, tensor_if->op, svtc_1.data,
          svtc_2.data, ret);
      break;
    case _NNS_INT16:
      operator_func (cv->data, int16_t, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_UINT16:
      operator_func (cv->data, uint16_t, tensor_if->op, svtc_1.data,
          svtc_2.data, ret);
      break;
    case _NNS_INT8:
      operator_func (cv->data, int8_t, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_UINT8:
      operator_func (cv->data, uint8_t, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_FLOAT64:
      operator_func (cv->data, double, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_FLOAT32:
      operator_func (cv->data, float, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_INT64:
      operator_func (cv->data, int64_t, tensor_if->op, svtc_1.data, svtc_2.data,
          ret);
      break;
    case _NNS_UINT64:
      operator_func (cv->data, uint64_t, tensor_if->op, svtc_1.data,
          svtc_2.data, ret);
      break;
    default:
      GST_ERROR_OBJECT (tensor_if, "Unknown tensor type %d", cv->type);
      return FALSE;
  }
  *result = ret;
  return TRUE;
}

#define get_double_val(type, in, out) do { \
    switch (type) { \
      case _NNS_INT32: out = (double) (*(int32_t *) in); break; \
      case _NNS_UINT32: out = (double) (*(uint32_t *) in); break; \
      case _NNS_INT16: out = (double) (*(int16_t *) in);break; \
      case _NNS_UINT16: out = (double) (*(uint16_t *) in); break; \
      case _NNS_INT8: out = (double) (*(int8_t *) in); break; \
      case _NNS_UINT8: out = (double) (*(uint8_t *) in); break; \
      case _NNS_FLOAT64: out = (double) (*(double *) in); break; \
      case _NNS_FLOAT32: out = (double) (*(float *) in); break; \
      case _NNS_INT64: out = (double) (*(int64_t *) in); break; \
      case _NNS_UINT64: out = (double) (*(uint64_t *) in); break; \
      default: g_assert (0); break; \
    } \
  } while (0)

/**
 * @brief Calculate average value of the nth tensor
 */
static void
gst_tensor_if_get_tensor_average (GstTensorIf * tensor_if,
    GstBuffer * buf, tensor_if_data_s * cv, gint nth)
{
  GstMemory *in_mem;
  GstMapInfo in_info;
  uint32_t i, size, dsize;
  const uint32_t *in_dim;
  double avg, val = 0.0, sum = 0.0;

  tensor_type type = tensor_if->in_config.info.info[nth].type;
  dsize = gst_tensor_get_element_size (type);

  in_dim = tensor_if->in_config.info.info[nth].dimension;
  size = in_dim[0] * in_dim[1] * in_dim[2] * in_dim[3];

  in_mem = gst_buffer_peek_memory (buf, nth);
  gst_memory_map (in_mem, &in_info, GST_MAP_READ);

  for (i = 0; i < size; i++) {
    get_double_val (type, &in_info.data[i * dsize], val);
    sum += val;
  }
  avg = sum / size;

  gst_memory_unmap (in_mem, &in_info);

  cv->type = _NNS_FLOAT64;
  cv->data._double = avg;

  gst_tensor_if_typecast_value (tensor_if, cv, type);
}

/**
 * @brief Calculate compared value
 */
static gboolean
gst_tensor_if_calculate_cv (GstTensorIf * tensor_if, GstBuffer * buf,
    tensor_if_data_s * cv)
{
  switch (tensor_if->cv) {
    case TIFCV_A_VALUE:
    {
      GstMemory *in_mem;
      GstMapInfo in_info;
      GList *list;
      uint32_t idx = 0, nth, i, offset = 1;
      tensor_dim target;
      const uint32_t *in_dim;

      if (g_list_length (tensor_if->cv_option) != 5) {
        GST_ERROR_OBJECT (tensor_if,
            " Please specify a proper 'compared-value-option' property, e.g., 0:1:2:3,0");
        return FALSE;
      }
      for (list = tensor_if->cv_option; list->next != NULL; list = list->next) {
        target[idx++] = GPOINTER_TO_INT (list->data);
      }

      nth = GPOINTER_TO_INT (list->data);
      if (gst_buffer_n_memory (buf) <= nth) {
        GST_ERROR_OBJECT (tensor_if, " index should be lower than buffer size");
        return FALSE;
      }
      cv->type = tensor_if->in_config.info.info[nth].type;

      in_dim = tensor_if->in_config.info.info[nth].dimension;

      in_mem = gst_buffer_peek_memory (buf, nth);
      gst_memory_map (in_mem, &in_info, GST_MAP_READ);

      /* Find data index for mem access */
      idx = target[0];
      for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
        offset *= in_dim[i - 1];
        idx += (target[i]) * offset;
      }

      idx *= gst_tensor_get_element_size (cv->type);

      gst_tensor_if_set_data (cv, (gpointer) & in_info.data[idx]);
      gst_memory_unmap (in_mem, &in_info);

      break;
    }
    case TIFCV_TENSOR_AVERAGE_VALUE:
    {
      uint32_t nth;
      if (g_list_length (tensor_if->cv_option) != 1) {
        GST_ERROR_OBJECT (tensor_if,
            " Please specify a proper 'compared-value-option' property, For TENSOR_AVERAGE_VALUE, specify only one tensor. Tensors is not supported.");
        return FALSE;
      }
      nth = GPOINTER_TO_INT (tensor_if->cv_option->data);
      if (gst_buffer_n_memory (buf) <= nth) {
        GST_ERROR_OBJECT (tensor_if, " index should be lower than buufer size");
        return FALSE;
      }
      gst_tensor_if_get_tensor_average (tensor_if, buf, cv, nth);
      break;
    }
    default:
      GST_ERROR_OBJECT (tensor_if,
          " Compared value is not supported yet or not defined");
      return FALSE;
  }
  return TRUE;
}

/**
 * @brief Determining whether a given condition is true or false
 * @param tensor_if TensorIf Object
 * @param buf gstbuffer from sink pad
 * @return return TRUE if no error
 */
static gboolean
gst_tensor_if_check_condition (GstTensorIf * tensor_if, GstBuffer * buf,
    gboolean * result)
{
  tensor_if_data_s cv = {.type = _NNS_END,.data._uint8_t = 0 };

  if (!gst_tensor_if_calculate_cv (tensor_if, buf, &cv)) {
    GST_ERROR_OBJECT (tensor_if, " failed to calculate compared value");
    return FALSE;
  }
  return gst_tensor_if_get_comparison_result (tensor_if, &cv, result);
}

/**
 * @brief chain function for sink (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_if_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  gint num_tensors, i;
  GstFlowReturn res = GST_FLOW_OK;
  GstTensorIf *tensor_if = GST_TENSOR_IF (parent);
  gboolean condition_result;
  tensor_if_behavior curr_act = TIFB_PASSTHROUGH;
  tensor_if_srcpads which_srcpad = TIFSP_THEN_PAD;
  GList *curr_act_option = NULL;
  GstTensorsConfig *config;
  GstTensorPad *srcpad;
  GstBuffer *outbuf = NULL;
  GstMemory *mem = NULL;
  gboolean created;
  GstClockTime ts;

  num_tensors = tensor_if->in_config.info.num_tensors;
  GST_DEBUG_OBJECT (tensor_if, " Number of Tensors: %d", num_tensors);
  /* supposed n memory blocks in buffer */
  g_assert (gst_buffer_n_memory (buf) == num_tensors);

  if (!gst_tensor_if_check_condition (tensor_if, buf, &condition_result)) {
    GST_ERROR_OBJECT (tensor_if, " Failed to check condition");
    return GST_FLOW_ERROR;
  }
  if (condition_result == TRUE) {
    curr_act = tensor_if->act_then;
    curr_act_option = tensor_if->then_option;
    which_srcpad = TIFSP_THEN_PAD;
  } else {
    curr_act = tensor_if->act_else;
    curr_act_option = tensor_if->else_option;
    which_srcpad = TIFSP_ELSE_PAD;
  }

  config = &tensor_if->out_config[which_srcpad];

  if (config->info.num_tensors == 0) {
    config->rate_n = tensor_if->in_config.rate_n;
    config->rate_d = tensor_if->in_config.rate_d;
  }

  switch (curr_act) {
    case TIFB_PASSTHROUGH:
      if (config->info.num_tensors == 0) {
        gst_tensors_info_copy (&config->info, &tensor_if->in_config.info);
      }
      outbuf = gst_buffer_ref (buf);

      break;
    case TIFB_TENSORPICK:
    {
      GList *list;
      gint info_idx = 0;

      outbuf = gst_buffer_new ();
      for (list = curr_act_option; list != NULL; list = list->next) {
        i = GPOINTER_TO_INT (list->data);
        if (config->info.num_tensors == 0) {
          gst_tensor_info_copy (&config->info.info[info_idx++],
              &tensor_if->in_config.info.info[i]);
        }
        mem = gst_buffer_get_memory (buf, i);
        gst_buffer_append_memory (outbuf, mem);
      }
      config->info.num_tensors = info_idx;
      break;
    }
    case TIFB_SKIP:
      goto done;
    default:
      GST_DEBUG_OBJECT (tensor_if, " Not defined behavior");
      break;
  }

  srcpad =
      gst_tensor_if_get_tensor_pad (tensor_if, config, &created, which_srcpad);

  if (created) {
    GstSegment segment;
    gst_segment_init (&segment, GST_FORMAT_TIME);
    gst_pad_push_event (srcpad->pad, gst_event_new_segment (&segment));
  }

  outbuf = gst_buffer_make_writable (outbuf);

  /* metadata from incoming buffer */
  gst_buffer_copy_into (outbuf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

  ts = GST_BUFFER_TIMESTAMP (buf);
  if (srcpad->last_ts == GST_CLOCK_TIME_NONE || srcpad->last_ts != ts) {
    srcpad->last_ts = ts;
  } else {
    GST_DEBUG_OBJECT (tensor_if, "invalid timestamp %" GST_TIME_FORMAT,
        GST_TIME_ARGS (ts));
  }

  res = gst_pad_push (srcpad->pad, outbuf);
  res = gst_tensor_if_combine_flows (tensor_if, srcpad, res);

done:
  gst_buffer_unref (buf);
  return res;
}
