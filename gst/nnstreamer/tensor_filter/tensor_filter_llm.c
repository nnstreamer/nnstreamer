#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <nnstreamer_util.h>
#include <string.h>
#include "tensor_filter_llm.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_filter_llm_debug);
#define GST_CAT_DEFAULT gst_tensor_filter_llm_debug

/**
 * @brief Template for sink pad.
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT ";"
        GST_TENSORS_CAP_MAKE ("{ flexible }")));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_filter_llm_parent_class parent_class
G_DEFINE_TYPE (GstTensorFilterLLM, gst_tensor_filter_llm, GST_TYPE_ELEMENT);

static void gst_tensor_filter_llm_finalize (GObject * object);
static void gst_tensor_filter_llm_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_llm_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstStateChangeReturn gst_tensor_filter_llm_change_state (GstElement *
    element, GstStateChange transition);
static gboolean gst_tensor_filter_llm_src_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_filter_llm_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static GstFlowReturn gst_tensor_filter_llm_chain (GstPad * sinkpad,
    GstObject * parent, GstBuffer * inbuf);

/**
 * @brief Initialize the tensor_filter_llm's class.
 */
static void
gst_tensor_filter_llm_class_init (GstTensorFilterLLMClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_filter_llm_debug, "tensor_filter_llm", 0,
      "Element to generate the tensors");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_filter_llm_set_property;
  object_class->get_property = gst_tensor_filter_llm_get_property;
  object_class->finalize = gst_tensor_filter_llm_finalize;

  element_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_llm_change_state);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_template));

  gst_element_class_set_static_metadata (element_class, "Tensor FilterLLM",
      "Filter/FilterLLM", "Element to filter_llm the tensors",
      "Samsung Electronics Co., Ltd.");
}

/**
 * @brief Initialize tensor_filter_llm element.
 */
static void
gst_tensor_filter_llm_init (GstTensorFilterLLM * self)
{
  /* setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_filter_llm_sink_event));
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /* setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_event_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_filter_llm_src_event));
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_filter_llm_chain));
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_filter_llm_finalize (GObject * object)
{
  GstTensorFilterLLM *self;
  self = GST_TENSOR_FILTER_LLM (object);

  gst_tensors_config_free (&self->in_config);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_filter_llm properties.
 */
static void
gst_tensor_filter_llm_set_property (GObject * object, guint prop_id,
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
 * @brief Getter for tensor_filter_llm properties.
 */
static void
gst_tensor_filter_llm_get_property (GObject * object, guint prop_id,
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
gst_tensor_filter_llm_change_state (GstElement * element,
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
gst_tensor_filter_llm_src_event (GstPad * pad, GstObject * parent,
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
gst_tensor_filter_llm_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstTensorFilterLLM *self;
  g_return_val_if_fail (event != NULL, FALSE);
  self = GST_TENSOR_FILTER_LLM (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstTensorsConfig config;
      GstStructure *structure;
      GstCaps *caps;
      gboolean ret;

      gst_event_parse_caps (event, &caps);
      ret = gst_tensors_config_from_cap (&self->in_config, caps);
      gst_event_unref (event);

      if (!ret) {
        ml_loge("Failted to parse caps from the event");
        return ret;
      }

      gst_tensors_config_init (&config);

      /* output tensor is always flexible */
      config.info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
      config.info.num_tensors = self->in_config.info.num_tensors;
      config.rate_d = self->in_config.rate_d;
      config.rate_n = self->in_config.rate_n;
      caps = gst_tensors_caps_from_config (&config);
      gst_pad_set_caps (self->srcpad, caps);
      gst_caps_unref (caps);

      return TRUE;
    }
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
gst_tensor_filter_llm_chain (GstPad * sinkpad, GstObject * parent,
    GstBuffer * inbuf)
{
  GstTensorFilterLLM *self;
  int count = 0;                /* Temporary variable to stop generating tensors */

  UNUSED (inbuf);
  UNUSED (sinkpad);

  self = GST_TENSOR_FILTER_LLM (parent);

  /** FIXME: Temporary condition to stop generating tensors */
  while (count++ < 10) {
    GstBuffer *dummy;
    GstMemory *mem;
    guint8 *data;
    GstCaps *caps;
    gsize mem_size;
    dummy = gst_buffer_new ();
    mem_size = 100; /** FIXME: Get mem_size from property */
    data = (guint8 *) g_malloc0 (mem_size);
    mem = gst_memory_new_wrapped (0, data, mem_size, 0, mem_size, data, g_free);

    gst_buffer_append_memory (dummy, mem);

    caps = gst_tensors_caps_from_config (&self->in_config);

    gst_pad_set_caps (self->srcpad, caps);
    gst_caps_unref (caps);
    gst_pad_push (self->srcpad, dummy);
  }

  return GST_FLOW_OK;
}
