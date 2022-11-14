/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-IF
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	gsttensor_if.c
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
 * Supplied_Value is a value from tensor-if properties.
 *
 * If the given if-condition is simple enough (e.g., if a specific element
 * is between a given range in a tensor frame), it can be expressed as:
 * <refsect2>
 * <title>Example launch line with simple if condition</title>
 * gst-launch ... (some tensor stream) !
 *      tensor_if name=tif
 *        compared-value=A_VALUE compared-value-option=3:4:2:5,0
 *        operator=RANGE_INCLUSIVE
 *        supplied-value=10,100
 *        then=PASSTHROUGH
 *        else=TENSORPICK
 *        else-option=1
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
 *      ! tensor_if compared-value=A_VALUE
 *          compared-value-option=0:0:0:0,0 # 1st tensor's [0][0][0][0].
 *          operator=EQ
 *          supplied-value=1
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

#include <nnstreamer_subplugin.h>
#include <nnstreamer_util.h>
#include "gsttensor_if.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!tensor_if->silent)
#endif

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
 * @brief A private function to register GEnumValue array for the 'compared-value' property
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
      {TIFCV_CUSTOM, "CUSTOM", "Decide based on a user defined callback"},
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
      "TensorIf",
      "Filter/Tensor",
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
  tensor_if->cv_option = NULL;
  tensor_if->then_option = NULL;
  tensor_if->else_option = NULL;
  memset (tensor_if->sv, 0, sizeof (tensor_if_sv_s) * 2);
  memset (&tensor_if->custom, 0, sizeof (custom_cb_s));
  tensor_if->custom_configured = FALSE;

  g_mutex_init (&tensor_if->lock);
}

/**
 * @brief function to remove srcpad list
 */
static void
gst_tensor_if_remove_src_pads (GstTensorIf * tensor_if)
{
  while (tensor_if->srcpads != NULL) {
    GstTensorPad *tensor_pad = tensor_if->srcpads->data;
    gst_element_remove_pad (GST_ELEMENT (tensor_if), tensor_pad->pad);
    g_free (tensor_pad);
    tensor_if->srcpads =
        g_slist_delete_link (tensor_if->srcpads, tensor_if->srcpads);
  }
  tensor_if->srcpads = NULL;
  tensor_if->num_srcpads = 0;
}

/**
 * @brief dispose function for tensor if (gst element vmethod)
 */
static void
gst_tensor_if_dispose (GObject * object)
{
  GstTensorIf *tensor_if = GST_TENSOR_IF (object);
  g_mutex_clear (&tensor_if->lock);

  gst_tensor_if_remove_src_pads (tensor_if);
  g_list_free (tensor_if->cv_option);
  g_list_free (tensor_if->then_option);
  g_list_free (tensor_if->else_option);
  g_free (tensor_if->custom.name);
  tensor_if->custom.func = NULL;
  tensor_if->custom.data = NULL;
  tensor_if->custom_configured = FALSE;

  G_OBJECT_CLASS (parent_class)->dispose (object);
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
 * @brief Convert GValue to GList for cv option
 */
static void
gst_tensor_if_set_property_cv_option (const GValue * value, GList ** prop_list)
{
  gint64 val;
  gint length, i;
  const gchar *param = g_value_get_string (value);
  gchar **strv = g_strsplit_set (param, ",", -1);
  GValue tmp = G_VALUE_INIT;

  length = g_strv_length (strv);

  if (length > 2) {
    ml_loge
        ("Invalid compared value option. It should be in the form of 'IDX_DIM0: ... :INDEX_DIM_LAST,nth-tensor'(A_VALUE) or 'nth-tensor' (TENSOR_AVERAGE_VALUE)");
    g_strfreev (strv);
    return;
  }

  g_value_init (&tmp, G_TYPE_STRING);
  g_value_set_string (&tmp, strv[0]);

  gst_tensor_if_set_property_glist (&tmp, prop_list, ":");

  /* A_VALUE */
  if (length == 2) {
    length = g_list_length (*prop_list);

    /* append zero value for undefined dimensions */
    for (i = length; i < NNS_TENSOR_RANK_LIMIT; i++) {
      *prop_list = g_list_append (*prop_list, GINT_TO_POINTER (0));
    }

    val = g_ascii_strtoll (strv[1], NULL, 10);
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

  if (!param) {
    ml_loge ("Invalid supplied value. The value is NULL.");
    return;
  }

  if (strchr (param, '.') || strchr (param, 'E') || strchr (param, 'e')) {
    is_float = TRUE;
  }

  sv->num = num;
  for (i = 0; i < num; i++) {
    if (is_float) {
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
 * @brief Set custom compared value property
 */
static void
gst_tensor_if_configure_custom_prop (GstTensorIf * self)
{
  if (!self->custom.name)
    return;

  if (self->cv == TIFCV_CUSTOM) {
    const custom_cb_s *ptr = get_subplugin (NNS_IF_CUSTOM, self->custom.name);
    if (!ptr) {
      nns_logw ("Failed to find custom subplugin of the tensor_if");
      return;
    }
    self->custom_configured = TRUE;
    self->custom.func = (*ptr).func;
    self->custom.data = (*ptr).data;
  }
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
      gst_tensor_if_configure_custom_prop (self);
      break;
    case PROP_CV_OPTION:
      g_free (self->custom.name);
      self->custom.name = g_value_dup_string (value);
      gst_tensor_if_configure_custom_prop (self);
      gst_tensor_if_set_property_cv_option (value, &self->cv_option);
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
  GPtrArray *arr;
  gchar **strings;
  guint len;

  if (prop_list == NULL) {
    g_value_set_string (value, "");
    return;
  }

  arr = g_ptr_array_new ();
  for (list = prop_list; list != NULL; list = list->next) {
    g_ptr_array_add (arr, g_strdup_printf ("%i", GPOINTER_TO_INT (list->data)));
  }
  g_ptr_array_add (arr, NULL);

  len = arr->len;

  if (prop_id == PROP_CV_OPTION && len % (NNS_TENSOR_RANK_LIMIT + 2) == 0) {
    gchar *dim;
    gchar *tensor = (gchar *) g_ptr_array_index (arr, len - 2);
    g_ptr_array_remove_index (arr, len - 2);
    strings = (gchar **) g_ptr_array_free (arr, FALSE);
    dim = g_strjoinv (":", strings);
    p = g_strjoin (",", dim, tensor, NULL);
    g_free (dim);
    g_free (tensor);
  } else {
    strings = (gchar **) g_ptr_array_free (arr, FALSE);
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
  guint i;
  gchar *p;
  GPtrArray *arr;
  gchar **strings;

  if (sv == NULL || sv->num == 0) {
    g_value_set_string (value, "");
    return;
  }

  arr = g_ptr_array_new ();
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
      if (self->cv == TIFCV_CUSTOM) {
        g_value_set_string (value, self->custom.name ? self->custom.name : "");
      } else {
        gst_tensor_if_property_to_string (value, self->cv_option, prop_id);
      }
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
 * @brief Checking if the source pad is created and if not, create TensorPad
 * @param tesnor_if TensorIf Object
 * @param config Tensors Config Data
 * @param nth source ordering
 * @return TensorPad if pad is already created, then return created pad.
 *         If not return new pad after creation.
 */
static GstTensorPad *
gst_tensor_if_get_tensor_pad (GstTensorIf * tensor_if,
    GstTensorsConfig * config, gboolean * created, guint nth)
{
  GSList *walk;
  GstPad *pad;
  GstTensorPad *tensorpad;
  gchar *name;
  GstCaps *caps = NULL;

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

  caps = gst_tensor_pad_caps_from_config (pad, config);

  silent_debug_caps (tensor_if, caps, "out caps");
  gst_pad_set_caps (pad, caps);

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
    tensor_data_s * cv, gboolean * result)
{
  gboolean ret = FALSE;
  tensor_data_s svtc_1, svtc_2;

  svtc_1.type = tensor_if->sv->type;
  svtc_1.data = tensor_if->sv->data[0];
  gst_tensor_data_typecast (&svtc_1, cv->type);

  svtc_2.type = tensor_if->sv->type;
  svtc_2.data = tensor_if->sv->data[1];
  gst_tensor_data_typecast (&svtc_2, cv->type);

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

/**
 * @brief Calculate average value of the nth tensor
 */
static gboolean
gst_tensor_if_get_tensor_average (GstTensorIf * tensor_if,
    GstBuffer * buf, tensor_data_s * cv, gint nth)
{
  GstMemory *in_mem;
  GstMapInfo in_info;
  gdouble *avg = NULL;
  tensor_type type = tensor_if->in_config.info.info[nth].type;

  in_mem = gst_buffer_peek_memory (buf, nth);
  if (!gst_memory_map (in_mem, &in_info, GST_MAP_READ)) {
    GST_WARNING_OBJECT (tensor_if, "Failed to map the input buffer.");
    return FALSE;
  }

  gst_tensor_data_raw_average (in_info.data, in_info.size, type, &avg);

  gst_memory_unmap (in_mem, &in_info);

  gst_tensor_data_set (cv, _NNS_FLOAT64, avg);
  gst_tensor_data_typecast (cv, type);

  g_free (avg);
  return TRUE;
}

/**
 * @brief Calculate compared value
 */
static gboolean
gst_tensor_if_calculate_cv (GstTensorIf * tensor_if, GstBuffer * buf,
    tensor_data_s * cv)
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
      tensor_type in_type;

      if (g_list_length (tensor_if->cv_option) != NNS_TENSOR_RANK_LIMIT + 1) {
        GST_ERROR_OBJECT (tensor_if,
            "Please specify a proper 'compared-value-option' property, e.g., 0:1:2:3,0");
        return FALSE;
      }
      for (list = tensor_if->cv_option; list->next != NULL; list = list->next) {
        target[idx++] = GPOINTER_TO_INT (list->data);
      }

      nth = GPOINTER_TO_INT (list->data);
      if (gst_buffer_n_memory (buf) <= nth) {
        GST_ERROR_OBJECT (tensor_if, "Index should be lower than buffer size");
        return FALSE;
      }

      in_type = tensor_if->in_config.info.info[nth].type;
      in_dim = tensor_if->in_config.info.info[nth].dimension;

      in_mem = gst_buffer_peek_memory (buf, nth);
      if (!gst_memory_map (in_mem, &in_info, GST_MAP_READ)) {
        GST_WARNING_OBJECT (tensor_if, "Failed to map the input buffer.");
        return FALSE;
      }

      /* Find data index for mem access */
      idx = target[0];
      for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
        offset *= in_dim[i - 1];
        idx += (target[i]) * offset;
      }

      idx *= gst_tensor_get_element_size (in_type);

      gst_tensor_data_set (cv, in_type, in_info.data + idx);
      gst_memory_unmap (in_mem, &in_info);

      break;
    }
    case TIFCV_TENSOR_AVERAGE_VALUE:
    {
      uint32_t nth;
      if (g_list_length (tensor_if->cv_option) != 1) {
        GST_ERROR_OBJECT (tensor_if,
            "Please specify a proper 'compared-value-option' property, For TENSOR_AVERAGE_VALUE, specify only one tensor. Tensors is not supported.");
        return FALSE;
      }
      nth = GPOINTER_TO_INT (tensor_if->cv_option->data);
      if (gst_buffer_n_memory (buf) <= nth) {
        GST_ERROR_OBJECT (tensor_if, "Index should be lower than buffer size");
        return FALSE;
      }
      return gst_tensor_if_get_tensor_average (tensor_if, buf, cv, nth);
    }
    default:
      GST_ERROR_OBJECT (tensor_if,
          "Compared value is not supported yet or not defined");
      return FALSE;
  }
  return TRUE;
}

/**
 * @brief Registers a callback for tensor_if custom condition
 * @return 0 if success. -ERRNO if error.
 */
int
nnstreamer_if_custom_register (const gchar * name, tensor_if_custom func,
    void *data)
{
  custom_cb_s *ptr;

  g_return_val_if_fail (name && strlen (name), -EINVAL);
  g_return_val_if_fail (func, -EINVAL);

  if (!(ptr = g_try_new0 (custom_cb_s, 1)))
    return -ENOMEM;

  ptr->func = func;
  ptr->data = data;

  if (register_subplugin (NNS_IF_CUSTOM, name, ptr))
    return 0;

  g_free (ptr);
  return -EINVAL;
}

/**
 * @brief Unregisters a callback for tensor_if custom condition
 * @return 0 if success. -ERRNO if error.
 */
int
nnstreamer_if_custom_unregister (const gchar * name)
{
  custom_cb_s *ptr;

  ptr = (custom_cb_s *) get_subplugin (NNS_IF_CUSTOM, name);
  if (!unregister_subplugin (NNS_IF_CUSTOM, name)) {
    ml_loge ("Failed to unregister custom callback %s.", name);
    return -EINVAL;
  }
  g_free (ptr);

  return 0;
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
  gboolean ret = FALSE;

  if (tensor_if->cv == TIFCV_CUSTOM) {
    GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT];
    GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
    GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT];
    guint i, j;

    if (!tensor_if->custom_configured) {
      nns_loge ("custom condition of the tensor_if is not configured.");
      return FALSE;
    }

    for (i = 0; i < tensor_if->in_config.info.num_tensors; i++) {
      in_mem[i] = gst_buffer_peek_memory (buf, i);
      if (!gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
        for (j = 0; j < i; j++)
          gst_memory_unmap (in_mem[j], &in_info[j]);
        GST_WARNING_OBJECT (tensor_if, "Cannot map input memory buffer(%d)\n",
            i);
        return FALSE;
      }
      in_tensors[i].data = in_info[i].data;
      in_tensors[i].size = in_info[i].size;
    }

    ret = tensor_if->custom.func (&tensor_if->in_config.info, in_tensors,
        tensor_if->custom.data, result);

    for (i = 0; i < tensor_if->in_config.info.num_tensors; i++)
      gst_memory_unmap (in_mem[i], &in_info[i]);
  } else {
    tensor_data_s cv = {.type = _NNS_END,.data._uint8_t = 0 };
    if (!gst_tensor_if_calculate_cv (tensor_if, buf, &cv)) {
      GST_ERROR_OBJECT (tensor_if, " failed to calculate compared value");
      return FALSE;
    }
    ret = gst_tensor_if_get_comparison_result (tensor_if, &cv, result);
  }

  return ret;
}

/**
 * @brief chain function for sink (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_if_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  guint num_tensors, i;
  GstFlowReturn res = GST_FLOW_OK;
  GstTensorIf *tensor_if = GST_TENSOR_IF (parent);
  gboolean condition_result = FALSE;
  tensor_if_behavior curr_act = TIFB_PASSTHROUGH;
  tensor_if_srcpads which_srcpad = TIFSP_THEN_PAD;
  GList *curr_act_option = NULL;
  GstTensorsConfig *config;
  GstTensorPad *srcpad;
  GstBuffer *outbuf = NULL;
  GstMemory *mem = NULL;
  gboolean created;
  GstClockTime ts;
  UNUSED (pad);

  num_tensors = tensor_if->in_config.info.num_tensors;
  GST_DEBUG_OBJECT (tensor_if, " Number of Tensors: %u", num_tensors);
  /* supposed n memory blocks in buffer */
  g_assert (gst_buffer_n_memory (buf) == num_tensors);

  if (!gst_tensor_if_check_condition (tensor_if, buf, &condition_result)) {
    GST_ERROR_OBJECT (tensor_if, " Failed to check condition");
    return GST_FLOW_ERROR;
  }

  if (condition_result) {
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
