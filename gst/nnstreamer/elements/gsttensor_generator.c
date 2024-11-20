#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <nnstreamer_util.h>
#include <string.h>
#include "gsttensor_generator.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_generator_debug);
#define GST_CAT_DEFAULT gst_tensor_generator_debug

/**
 * @brief Template for sink pad.
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT ";"
        GST_TENSORS_CAP_MAKE ("{ static, flexible }")));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_generator_parent_class parent_class
G_DEFINE_TYPE (GstTensorGenerator, gst_tensor_generator, GST_TYPE_ELEMENT);

static void gst_tensor_generator_finalize (GObject * object);
static void gst_tensor_generator_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_generator_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstStateChangeReturn gst_tensor_generator_change_state (GstElement *
    element, GstStateChange transition);
static gboolean gst_tensor_generator_src_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_generator_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static GstFlowReturn gst_tensor_generator_chain (GstPad * sinkpad,
    GstObject * parent, GstBuffer * inbuf);

/**
 * @brief Initialize the tensor_generator's class.
 */
static void
gst_tensor_generator_class_init (GstTensorGeneratorClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_generator_debug, "tensor_generator", 0,
      "Element to generate the tensors");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_generator_set_property;
  object_class->get_property = gst_tensor_generator_get_property;
  object_class->finalize = gst_tensor_generator_finalize;

  element_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_generator_change_state);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_template));

  gst_element_class_set_static_metadata (element_class, "Tensor Generator",
      "Filter/Generator", "Element to generator the tensors",
      "Samsung Electronics Co., Ltd.");
}

/**
 * @brief Initialize tensor_generator element.
 */
static void
gst_tensor_generator_init (GstTensorGenerator * self)
{
  /* setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_generator_sink_event));
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /* setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_event_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_generator_src_event));
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_generator_chain));
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_generator_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_generator properties.
 */
static void
gst_tensor_generator_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  UNUSED (value);
  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_generator properties.
 */
static void
gst_tensor_generator_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  UNUSED (value);
  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Handle state transition.
 */
static GstStateChangeReturn
gst_tensor_generator_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn ret;

  switch (transition) {
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    default:
      break;
  }

  return ret;
}

/**
 * @brief Handle event on src pad.
 */
static gboolean
gst_tensor_generator_src_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
      /* disable seeking */
      gst_event_unref (event);
      return FALSE;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief Handle event on sink pad.
 */
static gboolean
gst_tensor_generator_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
      /* disable seeking */
      gst_event_unref (event);
      return FALSE;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

static GstFlowReturn
gst_tensor_generator_chain (GstPad * sinkpad, GstObject * parent,
    GstBuffer * inbuf)
{
  UNUSED (sinkpad);
  UNUSED (parent);
  UNUSED (inbuf);

  ml_loge ("chain");

  return GST_FLOW_OK;
}
