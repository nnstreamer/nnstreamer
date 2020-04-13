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
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

#define gst_tensor_if_parent_class parent_class
G_DEFINE_TYPE (GstTensorIf, gst_tensor_if, GST_TYPE_BASE_TRANSFORM);

/* GObject vmethod implementations */
static void gst_tensor_if_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_if_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_if_finalize (GObject * object);

/* GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensor_if_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstCaps *gst_tensor_if_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * _if);
static GstCaps *gst_tensor_if_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_if_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_if_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
static gboolean gst_tensor_if_start (GstBaseTransform * trans);
static gboolean gst_tensor_if_stop (GstBaseTransform * trans);
static gboolean gst_tensor_if_sink_event (GstBaseTransform * trans,
    GstEvent * event);

static void gst_tensor_if_install_properties (GObjectClass * gobject_class);

/**
 * @brief initialize the tensor_if's class (GST Standard)
 */
static void
gst_tensor_if_class_init (GstTensorIfClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_if_debug, "tensor_if", 0,
      "Tensor if to control streams based on tensor(s) values");

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_if_set_property;
  gobject_class->get_property = gst_tensor_if_get_property;
  gobject_class->finalize = gst_tensor_if_finalize;

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

  /* Refer: https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;
  /**
   * Tensor-IF always have the same caps on src/sink; however,
   * it won't pass-through the data
   */

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_if_transform);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_if_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_if_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_if_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_if_transform_size);

  /* setup sink event */
  trans_class->sink_event = GST_DEBUG_FUNCPTR (gst_tensor_if_sink_event);

  /* start/stop to call open/close */
  trans_class->start = GST_DEBUG_FUNCPTR (gst_tensor_if_start);
  trans_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_if_stop);
}

/**
 * @brief initialize the new element (GST Standard)
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_if_init (GstTensorIf * self)
{
  /** @todo NYI: initialize values of self */
  self->silent = TRUE;
}

/**
 * @brief Function to finalize instance. (GST Standard)
 */
static void
gst_tensor_if_finalize (GObject * object)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */

  /** @todo NYI: finialize (free-up) everything in self */

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_if properties.
 */
static void
gst_tensor_if_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */
  /** @todo NYI! */
}

/**
 * @brief Getter for tensor_if properties.
 */
static void
gst_tensor_if_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */
  /** @todo NYI! */
}

/**
 * @brief non-ip transform. required vmethod of GstBaseTransform.
 */
static GstFlowReturn
gst_tensor_if_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */

  return GST_FLOW_ERROR; /** @todo NYI! */
}

/**
 * @brief configure tensor-srcpad cap from "proposed" cap. (GST Standard)
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap (if direction GST_PAD_SINK)
 * @if this element's cap (don't know specifically.)
 *
 * Be careful not to fix/set caps at this stage. Negotiation not completed yet.
 */
static GstCaps *
gst_tensor_if_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  /** @todo : SRC and SINK caps are identical! */
  return NULL; /** @todo NYI! */
}


/**
 * @brief fixate caps. required vmethod of GstBaseTransform.
 */
static GstCaps *
gst_tensor_if_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  /** @todo : SRC and SINK caps are identical! */
  return NULL; /** @todo NYI! */
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_if_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  /** @todo : SRC and SINK caps are identical! */
  return FALSE; /** @todo NYI! */
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_if_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */

  /** @todo : SRC and SINK caps are identical! */
  return FALSE; /** @todo NYI! */
}

/**
 * @brief Event handler for sink pad of tensor if.
 * @param trans "this" pointer
 * @param event a passed event object
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_if_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */

  switch (GST_EVENT_TYPE (event)) {
    /** @todo NYI! */
    default:
      break;
  }

  /** other events are handled in the default event handler */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * @brief Called when the element starts processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_if_start (GstBaseTransform * trans)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */

  return FALSE; /** @todo NYI! Do not allow to start! */
}

/**
 * @brief Called when the element stops processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_if_stop (GstBaseTransform * trans)
{
  /* GstTensorIf *self = GST_TENSOR_IF (object); */

  return TRUE; /** @todo NYI! but, let's allow to stop! */
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
}
