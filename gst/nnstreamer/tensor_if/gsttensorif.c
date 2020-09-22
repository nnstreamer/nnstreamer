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
 *      tensor_if compared_value=A_VALUE compared_value_option=3:4:2:5,0
 *      operator=RANGE_INCLUSIVE
 *      supplied_values=10,100
 *      then=PASSTHROUGH
 *      else=FILL_WITH_FILE else_option=${path_to_file}
 *    ! (tensor stream) ...
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

#include "gsttensorif.h"

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
  PROP_SV_OPTION, /**< Supplied Value Option */
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
      {TIFCV_A_VALUE, "A_VALUE", "a_value"},
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
  gst_tensors_config_init (&tensor_if->out_config);

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
 * @brief Convert GValue to GList according to delimiters
 */
static GList *
gst_tensor_if_set_property_glist (const GValue * value, GList * prop_list,
    const gchar *delimiters)
{
  gint i;
  gint64 val;
  const gchar *param = g_value_get_string (value);
  gchar **strv = g_strsplit_set (param, delimiters, -1);
  gint num = g_strv_length (strv);
  g_critical ("option num : %d", num);
  for (i = 0; i < num; i++) {
    val = g_ascii_strtoll (strv[i], NULL, 10);
    g_critical ("cv option %d th option : %ld", i, val);
    prop_list =
        g_list_append (prop_list, GINT_TO_POINTER (val));
  }
  g_strfreev (strv);

  return prop_list;
}

/**
 * @brief Setter for tensor_if properties.
 */
static void
gst_tensor_if_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorIf *self = GST_TENSOR_IF (object);

  g_critical ("Setting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_CV:
      self->cv = g_value_get_enum (value);
      g_critical ("Compared value : %d", self->cv);
      break;
    case PROP_CV_OPTION:
    {
      self->cv_option = gst_tensor_if_set_property_glist (value, self->cv_option, ":,");
      break;
    }
    case PROP_OP:
      self->op = g_value_get_enum (value);
      g_critical ("operator : %d", self->op);
      break;
    case PROP_SV:
    {
      self->sv = gst_tensor_if_set_property_glist (value, self->sv, ",");
      break;
    }
    case PROP_SV_OPTION:
      break;
    case PROP_THEN:
      self->act_then = g_value_get_enum (value);
      g_critical ("Set act_then = %d", self->act_then);
      break;
    case PROP_THEN_OPTION:
      self->then_option = gst_tensor_if_set_property_glist (value, self->then_option, ",");
      break;
    case PROP_ELSE:
      self->act_else = g_value_get_enum (value);
      g_critical ("Set act_else = %d", self->act_else);
      break;
    case PROP_ELSE_OPTION:
      self->else_option = gst_tensor_if_set_property_glist (value, self->else_option, ",");
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      g_critical ("Set silent = %d", self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_if properties.
 */
static void
gst_tensor_if_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorIf *self = GST_TENSOR_IF (object);
  g_critical ("Getting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_CV:
      g_value_set_enum (value, self->cv);
      break;
    case PROP_CV_OPTION:
      break;
    case PROP_OP:
      break;
    case PROP_SV:
      break;
    case PROP_SV_OPTION:
      break;
    case PROP_THEN:
      break;
    case PROP_THEN_OPTION:
      break;
    case PROP_ELSE:
      break;
    case PROP_ELSE_OPTION:
      break;
    case PROP_SILENT:
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
    g_param_spec_enum ("compared-value", "CV", "Compared value from input tensor(s)",
        GST_TYPE_TENSOR_IF_CV, TIFCV_A_VALUE,
        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
        GST_TYPE_TENSOR_IF_ACT, TIFB_PASSTHROUGH,
        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ELSE_OPTION,
    g_param_spec_string ("else-option", "THEN_OPTION",
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
      gst_tensor_if_parse_caps (tensor_if, caps);
      break;
    }
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}


/**
 * @brief Get tensor config info from configured tensors
 * @param tensor_if "this" pointer
 * @param config tensor config to be filled
 * @param index index of configured tensors
 * @return
 */
static gboolean
gst_tensor_if_get_tensor_config (GstTensorIf * tensor_if,
    GstTensorConfig * config, guint index)
{
  GstTensorsConfig *tensors_info;

  g_return_val_if_fail (tensor_if != NULL, FALSE);
  g_return_val_if_fail (config != NULL, FALSE);

  gst_tensor_config_init (config);

  tensors_info = &tensor_if->in_config;
  g_return_val_if_fail (index < tensors_info->info.num_tensors, FALSE);

  config->info = tensors_info->info.info[index];
  config->rate_n = tensors_info->rate_n;
  config->rate_d = tensors_info->rate_d;

  return TRUE;
}

/**
 * @brief Checking if the source pad is created and if not, create TensorPad
 * @param tesnor_if TensorIf Object
 * @param[out] created will be updated in this function
 * @param nth source ordering
 * @return TensorPad if pad is already created, then return created pad.
 *         If not return new pad after creation.
 */
static GstTensorPad *
gst_tensor_if_get_tensor_pad (GstTensorIf * tensor_if,
    gboolean * created, gint nth)
{
  GSList *walk;
  GstPad *pad;
  GstTensorPad *tensorpad;
  gchar *name;
  GstEvent *event;
  gchar *stream_id;
  GstCaps *caps;
  GstTensorConfig config;

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

  name = g_strdup_printf ("src_%u", tensor_if->num_srcpads);
  pad = gst_pad_new_from_static_template (&src_factory, name);
  g_free (name);

  tensorpad->pad = pad;
  tensorpad->nth = nth;
  tensorpad->last_ret = GST_FLOW_OK;
  tensorpad->last_ts = GST_CLOCK_TIME_NONE;

  tensor_if->srcpads = g_slist_append (tensor_if->srcpads, tensorpad);
  gst_tensor_if_get_tensor_config (tensor_if, &config, nth);

  tensor_if->num_srcpads++;

  gst_pad_use_fixed_caps (pad);
  gst_pad_set_active (pad, TRUE);


  if (!tensor_if->have_group_id) {
    event =
        gst_pad_get_sticky_event (tensor_if->sinkpad, GST_EVENT_STREAM_START,
        0);
    if (event) {
      tensor_if->have_group_id =
          gst_event_parse_group_id (event, &tensor_if->group_id);
      gst_event_unref (event);
    } else if (!tensor_if->have_group_id) {
      tensor_if->have_group_id = TRUE;
      tensor_if->group_id = gst_util_group_id_next ();
    }
  }

  stream_id =
      gst_pad_create_stream_id (pad, GST_ELEMENT_CAST (tensor_if),
      "other/tensors");

  event = gst_event_new_stream_start (stream_id);
  if (tensor_if->have_group_id)
    gst_event_set_group_id (event, tensor_if->group_id);

  gst_pad_store_sticky_event (pad, event);
  g_free (stream_id);
  gst_event_unref (event);

  caps = gst_tensor_caps_from_config (&config);
  gst_pad_set_caps (pad, caps);
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_if), pad);

  gst_caps_unref (caps);

  if (created) {
    *created = TRUE;
  }

  if (tensor_if->then_option != NULL) {
    GST_DEBUG_OBJECT (tensor_if, "TensorPick is set! : %dth tensor\n", nth);
    if (g_list_length (tensor_if->then_option) == tensor_if->num_srcpads) {
      gst_element_no_more_pads (GST_ELEMENT_CAST (tensor_if));
    }
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
 * @brief chain function for sink (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_if_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  gint num_tensors, i;
  GstFlowReturn res = GST_FLOW_OK;
  GstTensorIf *tensor_if;
  tensor_if = GST_TENSOR_IF (parent);

  num_tensors = tensor_if->in_config.info.num_tensors;
  GST_DEBUG_OBJECT (tensor_if, " Number of Tensors: %d", num_tensors);

  /* supposed n memory blocks in buffer */
  g_assert (gst_buffer_n_memory (buf) == num_tensors);

  for (i = 0; i < num_tensors; i++) {
    GstTensorPad *srcpad;
    GstBuffer *outbuf;
    GstMemory *mem;
    gboolean created;
    GstClockTime ts;

    if (tensor_if->then_option != NULL) {
      gboolean found = FALSE;
      GList *list;
      for (list = tensor_if->then_option; list != NULL; list = list->next) {
        if (i == GPOINTER_TO_INT (list->data)) {
          found = TRUE;
          break;
        }
      }
      if (!found)
        continue;
    }

    srcpad = gst_tensor_if_get_tensor_pad (tensor_if, &created, i);

    outbuf = gst_buffer_new ();
    mem = gst_buffer_get_memory (buf, i);
    gst_buffer_append_memory (outbuf, mem);
    ts = GST_BUFFER_TIMESTAMP (buf);

    if (created) {
      GstSegment segment;
      gst_segment_init (&segment, GST_FORMAT_TIME);
      gst_pad_push_event (srcpad->pad, gst_event_new_segment (&segment));
    }

    outbuf = gst_buffer_make_writable (outbuf);

    /* metadata from incoming buffer */
    gst_buffer_copy_into (outbuf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

    if (srcpad->last_ts == GST_CLOCK_TIME_NONE || srcpad->last_ts != ts) {
      srcpad->last_ts = ts;
    } else {
      GST_DEBUG_OBJECT (tensor_if, "invalid timestamp %" GST_TIME_FORMAT,
          GST_TIME_ARGS (ts));
    }

    GST_DEBUG_OBJECT (tensor_if,
        "pushing buffer with timestamp %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)));
    res = gst_pad_push (srcpad->pad, outbuf);
    res = gst_tensor_if_combine_flows (tensor_if, srcpad, res);

    if (res != GST_FLOW_OK)
      break;
  }

  gst_buffer_unref (buf);
  return res;
}
