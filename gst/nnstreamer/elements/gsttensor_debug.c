/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer tensor_debug
 * Copyright (C) 2022 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	gsttensor_debug.c
 * @date	23 Sep 2022
 * @brief	GStreamer plugin to help debug tensor streams.
 *
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_debug
 *
 * A filter that generates debug messages for developer at the insertion
 * point of the given pipeline. An application writer using an nnstreamer
 * pipeline can use tensor_debug to debug or get profile information in their
 * applications.
 *
 * Note that this does not support other/tensor, but only supports other/tensors.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_debug output-method=console-info capability=always ! tensor_sink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "gsttensor_debug.h"
#include "tensor_meta.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

GST_DEBUG_CATEGORY_STATIC (gst_tensor_debug_debug);
#define GST_CAT_DEFAULT gst_tensor_debug_debug

/**
 * This is a new element created after the obsoletion of other/tensor.
 * Use other/tensors if you want to use tensor_debug
 */
#define CAPS_STRING GST_TENSORS_CAP_MAKE(GST_TENSOR_FORMAT_ALL)

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

/**
 * @brief tensor_debug properties
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_OUTPUT,
  PROP_CAP,
  PROP_META,
};

#define C_FLAGS(v) ((guint) v)

#define TENSOR_DEBUG_TYPE_OUTPUT_FLAGS (tensor_debug_output_flags_get_type())
/**
 * @brief Flags for output_mode of GstTensorDebug
 */
static GType
tensor_debug_output_flags_get_type (void)
{
  static GType type = G_TYPE_INVALID;

  if (type == G_TYPE_INVALID) {
    static const GFlagsValue values[] = {
      {C_FLAGS (TDBG_OUTPUT_DISABLED),
            "Disable log output and write. Do not add other flags to have this flag effective.",
          "disabled"},
      {C_FLAGS (TDBG_OUTPUT_CONSOLE_I),
            "Console output with info. Cannot combine with other console flags",
          "console-info"},
      {C_FLAGS (TDBG_OUTPUT_CONSOLE_W),
            "Console output with warning. Cannot combine with other console flags",
          "console-warn"},
      {C_FLAGS (TDBG_OUTPUT_CONSOLE_E),
            "Console output with error. Cannot combine with other console flags",
          "console-error"},
      {C_FLAGS (TDBG_OUTPUT_GSTDBG_I),
            "Gstlog output with info. Cannot combine with other gstdbg flags",
          "gstdebug-info"},
      {C_FLAGS (TDBG_OUTPUT_GSTDBG_W),
            "Gstlog output with warning. Cannot combine with other gstdbg flags",
          "gstdebug-warn"},
      {C_FLAGS (TDBG_OUTPUT_GSTDBG_E),
            "Gstlog output with error. Cannot combine with other gstdbg flags",
          "gstdebug-error"},
      {C_FLAGS (TDBG_OUTPUT_CIRCULARBUF),
            "Store at gsttensor_debug circular buffer so that it can be retrieved by the application later (NYI)",
          "circularbuf"},
      {C_FLAGS (TDBG_OUTPUT_FILEWRITE),
          "Write to a file (NYI)", "filewrite"},
      {0, NULL, NULL}
    };
    type = g_flags_register_static ("gtd_output", values);
  }

  return type;
}

#define DEFAULT_TENSOR_DEBUG_OUTPUT_FLAGS (TDBG_OUTPUT_CONSOLE_I)

#define TENSOR_DEBUG_TYPE_CAPS (tensor_debug_cap_get_type())
/**
 * @brief Enums for cap_mode of GstTensorDebug
 */
static GType
tensor_debug_cap_get_type (void)
{
  static GType type = G_TYPE_INVALID;
  if (type == G_TYPE_INVALID) {
    static GEnumValue values[] = {
      {TDBG_CAP_DISABLED, "disabled", "Do not log stream capability"},
      {TDBG_CAP_SHOW_UPDATE, "updates",
          "Log stream capability if it is updated or initialized."},
      {TDBG_CAP_SHOW_UPDATE_F, "updates-full",
          "Log stream capability if the capability or dimensions of flexible/sparse tensors are updated. Logs dimension info of flexible/sparse tensors as well."},
      {TDBG_CAP_SHOW_ALWAYS, "always",
          "Always, log stream capability and tensor dimension information."},
      {0, NULL, NULL}
    };
    type = g_enum_register_static ("gtd_cap", values);
  }
  return type;
}

#define DEFAULT_TENSOR_DEBUG_CAP (TDBG_CAP_SHOW_UPDATE_F)

#define TENSOR_DEBUG_TYPE_META_FLAGS (tensor_debug_meta_flags_get_type())
/**
 * @brief Flags for meta_mode of GstTensorDebug
 */
static GType
tensor_debug_meta_flags_get_type (void)
{
  static GType type = G_TYPE_INVALID;

  if (type == G_TYPE_INVALID) {
    static const GFlagsValue values[] = {
      {C_FLAGS (TDBG_META_DISABLED),
          "Do not log stream metadata.", "disabled"},
      {C_FLAGS (TDBG_META_TIMESTAMP), "Log timestamp information", "timestamp"},
      {C_FLAGS (TDBG_META_QUERYSERVER),
          "Log tensor-query-server related information", "queryserver"},
      {0, NULL, NULL}
    };
    type = g_flags_register_static ("gtd_meta", values);
  }
  return type;
}

#define DEFAULT_TENSOR_DEBUG_META_FLAGS (TDBG_META_DISABLED)

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

#define gst_tensor_debug_parent_class parent_class
G_DEFINE_TYPE (GstTensorDebug, gst_tensor_debug, GST_TYPE_BASE_TRANSFORM);

/* gobject vmethods */
static void gst_tensor_debug_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_debug_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_debug_finalize (GObject * object);

/* gstbasetransform vmethods */
static GstFlowReturn gst_tensor_debug_transform_ip (GstBaseTransform * trans,
    GstBuffer * buffer);
static GstCaps *gst_tensor_debug_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_debug_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);

/**
 * @brief Initialize the tensor_debug's class.
 */
static void
gst_tensor_debug_class_init (GstTensorDebugClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_debug_debug, "tensor_debug", 0,
      "Element to provide debug information of other/tensors streams. If this is enabled, the pipeline performance and overhead may be deteriorated significantly.");

  trans_class = (GstBaseTransformClass *) klass;
  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  /* GObjectClass vmethods */
  object_class->set_property = gst_tensor_debug_set_property;
  object_class->get_property = gst_tensor_debug_get_property;
  object_class->finalize = gst_tensor_debug_finalize;

  /**
   * GstTensorDebug::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorDebug::output:
   *
   * The combination of enums configuring output methods.
   * @todo check the behavior of name and nick (output methods vs output)
   */
  g_object_class_install_property (object_class, PROP_OUTPUT,
      g_param_spec_flags ("output-method", "output",
          "Output methods for debug/profile contents. Different methods can be enabled simultaneously.",
          TENSOR_DEBUG_TYPE_OUTPUT_FLAGS, DEFAULT_TENSOR_DEBUG_OUTPUT_FLAGS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorDebug::cap:
   *
   * The logging preference of the stream capability (GSTCAP).
   */
  g_object_class_install_property (object_class, PROP_CAP,
      g_param_spec_enum ("capability", "cap",
          "The logging preference for stream capability (GSTCAP)",
          TENSOR_DEBUG_TYPE_CAPS, DEFAULT_TENSOR_DEBUG_CAP,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorDebug::meta:
   *
   * The logging preference of in-stream metadata (GSTMETA).
   */
  g_object_class_install_property (object_class, PROP_META,
      g_param_spec_flags ("metadata", "meta",
          "The logging preference for stream metadata (GstMeta)",
          TENSOR_DEBUG_TYPE_META_FLAGS, DEFAULT_TENSOR_DEBUG_META_FLAGS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


  /* set pad template */
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_factory));

  gst_element_class_set_static_metadata (element_class,
      "TensorDebug",
      "Filter/Tensor",
      "Help debug or profile a tensor stream by logging the desired details of other/tensors. Users may log the details to console, files, or memory buffers.",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  /* GstBaseTransform vmethods */
  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_tensor_debug_transform_ip);

  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_debug_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_debug_set_caps);

  /* GstBaseTransform Property */
  trans_class->passthrough_on_same_caps = TRUE;
      /** This won't modify the contents! */
  trans_class->transform_ip_on_passthrough = TRUE;
      /** call transform_ip although it's passthrough */

  /**
   * Note.
   * Without transform_caps and with passthrough_on_same_caps = TRUE,
   * This element is not allowed to touch the contents, but can inspect
   * the contents with transform_ip by setting transform_ip_on_passthrough.
   */
}

/**
 * @brief Initialize tensor_debug element.
 */
static void
gst_tensor_debug_init (GstTensorDebug * self)
{
  /** init properties */
  self->silent = DEFAULT_SILENT;
  self->output_mode = DEFAULT_TENSOR_DEBUG_OUTPUT_FLAGS;
  self->cap_mode = DEFAULT_TENSOR_DEBUG_CAP;
  self->meta_mode = DEFAULT_TENSOR_DEBUG_META_FLAGS;

}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_debug_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_debug properties.
 */
static void
gst_tensor_debug_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDebug *self = GST_TENSOR_DEBUG (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      silent_debug (self, "Set silent = %d", self->silent);
      break;
    case PROP_OUTPUT:
      self->output_mode = g_value_get_flags (value);
      silent_debug (self, "Set output = %x", self->output_mode);
      break;
    case PROP_CAP:
      self->cap_mode = g_value_get_enum (value);
      silent_debug (self, "Set cap = %x", self->cap_mode);
      break;
    case PROP_META:
      self->meta_mode = g_value_get_flags (value);
      silent_debug (self, "Set meta = %x", self->meta_mode);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_debug properties.
 */
static void
gst_tensor_debug_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDebug *self = GST_TENSOR_DEBUG (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_OUTPUT:
      g_value_set_flags (value, self->output_mode);
      break;
    case PROP_CAP:
      g_value_set_enum (value, self->cap_mode);
      break;
    case PROP_META:
      g_value_set_flags (value, self->meta_mode);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief The core function that provides debug output based
 *        on the contents.
 */
static void
_gst_tensor_debug_output (GstTensorDebug * self, GstBuffer * buffer)
{
  UNUSED (self);
  UNUSED (buffer);
  /** @todo NYI: do the debug task */
}

/**
 * @brief in-place transform
 */
static GstFlowReturn
gst_tensor_debug_transform_ip (GstBaseTransform * trans, GstBuffer * buffer)
{
  GstTensorDebug *self = GST_TENSOR_DEBUG (trans);

  _gst_tensor_debug_output (self, buffer);

  return GST_FLOW_OK;
}

/**
 * @brief fixate caps. required vmethod of GstBaseTransform.
 */
static GstCaps *
gst_tensor_debug_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  UNUSED (trans);
  UNUSED (direction);
  UNUSED (caps);

  return gst_caps_fixate (othercaps);
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_debug_set_caps (GstBaseTransform * trans,
    GstCaps * in_caps, GstCaps * out_caps)
{
  UNUSED (trans);

  return gst_caps_can_intersect (in_caps, out_caps);
}
