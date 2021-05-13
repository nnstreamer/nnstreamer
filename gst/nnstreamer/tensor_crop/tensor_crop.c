/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file	tensor_crop.c
 * @date	10 May 2021
 * @brief	GStreamer element to crop the regions of incoming tensor
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_crop
 *
 * tensor_crop is a GStreamer element to crop the regions of incoming tensor.
 *
 * tensor_crop has two always sink pads - raw and info.
 * The raw pad accepts tensor (other/tensor) which will be cropped with crop info.
 * The info pad has capability for flexible tensor stream (other/tensors-flexible), that can have a various buffer size for crop info.
 * Incoming buffer on info pad should be an array of GstTensorCropInfo (see tensor_typedef.h).
 * Note that NNStreamer supports maximum 16 (NNS_TENSOR_SIZE_LIMIT) memory blocks in a buffer.
 * So, when incoming buffer on info pad has more than 16 GstTensorCropInfo array, tensor_crop will ignore the data and output buffer will have 16 memory blocks.
 *
 * The output is always in the format of other/tensors-flexible.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 tensor_crop name=crop ! (cropped tensors) ... \
 *     videotestsrc ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tee name=t \
 *       t. ! queue ! crop.raw \
 *       t. ! queue ! (process raw video tensor and push buffer which includes crop info) ! crop.info
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_crop.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_crop_debug);
#define GST_CAT_DEFAULT gst_tensor_crop_debug

/**
 * @brief tensor_crop properties
 */
enum
{
  PROP_0,
  PROP_SILENT
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Template for sink pad (raw data).
 */
static GstStaticPadTemplate raw_template = GST_STATIC_PAD_TEMPLATE ("raw",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

/**
 * @brief Template for sink pad (crop info).
 */
static GstStaticPadTemplate info_template = GST_STATIC_PAD_TEMPLATE ("info",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_crop_parent_class parent_class
G_DEFINE_TYPE (GstTensorCrop, gst_tensor_crop, GST_TYPE_ELEMENT);

static void gst_tensor_crop_finalize (GObject * object);
static void gst_tensor_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstStateChangeReturn gst_tensor_crop_change_state (GstElement * element,
    GstStateChange transition);
static gboolean gst_tensor_crop_src_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static gboolean gst_tensor_crop_sink_event (GstCollectPads * pads,
    GstCollectData * data, GstEvent * event, gpointer user_data);
static GstFlowReturn gst_tensor_crop_collected (GstCollectPads * pads,
    gpointer user_data);

/**
 * @brief Initialize the tensor_crop's class.
 */
static void
gst_tensor_crop_class_init (GstTensorCropClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_crop_debug, "tensor_crop", 0,
      "Element to crop the regions of incoming tensor");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_crop_set_property;
  object_class->get_property = gst_tensor_crop_get_property;
  object_class->finalize = gst_tensor_crop_finalize;

  /**
   * GstTensorCrop::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  element_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_crop_change_state);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&raw_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&info_template));

  gst_element_class_set_static_metadata (element_class,
      "TensorCrop",
      "Filter/Tensor",
      "Element to crop the regions of incoming tensor",
      "Samsung Electronics Co., Ltd.");
}

/**
 * @brief Clear and reset old pad data.
 */
static void
gst_tensor_crop_pad_reset (GstTensorCropPadData * cpad)
{
  gst_tensors_info_free (&cpad->config.info);
  gst_tensors_config_init (&cpad->config);
}

/**
 * @brief Clear and reset old data in tensor_crop.
 */
static void
gst_tensor_crop_reset (GstTensorCrop * self)
{
  GstTensorCropPadData *cpad;
  GSList *walk;

  if (self->collect) {
    walk = self->collect->data;

    while (walk) {
      cpad = (GstTensorCropPadData *) walk->data;

      gst_tensor_crop_pad_reset (cpad);
      walk = g_slist_next (walk);
    }
  }

  self->send_stream_start = TRUE;
}

/**
 * @brief Initialize tensor_crop element.
 */
static void
gst_tensor_crop_init (GstTensorCrop * self)
{
  /* setup sink pad */
  self->sinkpad_raw = gst_pad_new_from_static_template (&raw_template, "raw");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad_raw);

  self->sinkpad_info =
      gst_pad_new_from_static_template (&info_template, "info");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad_info);

  self->collect = gst_collect_pads_new ();
  gst_collect_pads_set_function (self->collect,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_collected), self);
  gst_collect_pads_set_event_function (self->collect,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_sink_event), self);

  gst_collect_pads_add_pad (self->collect, self->sinkpad_raw,
      sizeof (GstTensorCropPadData), NULL, TRUE);
  gst_collect_pads_add_pad (self->collect, self->sinkpad_info,
      sizeof (GstTensorCropPadData), NULL, TRUE);

  /* setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_event_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_src_event));
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /* init properties */
  self->silent = DEFAULT_SILENT;
  self->send_stream_start = TRUE;
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_crop_finalize (GObject * object)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  gst_tensor_crop_reset (self);

  if (self->collect) {
    gst_object_unref (self->collect);
    self->collect = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_crop properties.
 */
static void
gst_tensor_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_crop properties.
 */
static void
gst_tensor_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Handle state transition.
 */
static GstStateChangeReturn
gst_tensor_crop_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorCrop *self;
  GstStateChangeReturn ret;

  self = GST_TENSOR_CROP (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_collect_pads_start (self->collect);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_collect_pads_stop (self->collect);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_tensor_crop_reset (self);
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Handle event on src pad.
 */
static gboolean
gst_tensor_crop_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
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
gst_tensor_crop_sink_event (GstCollectPads * pads, GstCollectData * data,
    GstEvent * event, gpointer user_data)
{
  GstTensorCropPadData *cpad;

  g_return_val_if_fail (event != NULL, FALSE);

  cpad = (GstTensorCropPadData *) data;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      GstStructure *structure;

      gst_event_parse_caps (event, &caps);
      structure = gst_caps_get_structure (caps, 0);

      gst_tensors_config_from_structure (&cpad->config, structure);

      gst_event_unref (event);
      return gst_tensors_config_validate (&cpad->config);
    }
    default:
      break;
  }

  return gst_collect_pads_event_default (pads, data, event, FALSE);
}

/**
 * @brief Set pad caps if not negotiated.
 */
static GstFlowReturn
gst_tensor_crop_negotiate (GstTensorCrop * self)
{
  if (!gst_pad_has_current_caps (self->sinkpad_raw)) {
    GST_ERROR_OBJECT (self,
        "The raw pad of tensor_crop '%s' does not have pad caps.",
        GST_ELEMENT_NAME (self));
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (!gst_pad_has_current_caps (self->sinkpad_info)) {
    GST_ERROR_OBJECT (self,
        "The info pad of tensor_crop '%s' does not have pad caps.",
        GST_ELEMENT_NAME (self));
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (!gst_pad_has_current_caps (self->srcpad)) {
    GstCaps *caps;
    GstSegment segment;

    if (self->send_stream_start) {
      gchar *sid;

      sid = g_strdup_printf ("%s-%08x",
          GST_ELEMENT_NAME (self), g_random_int ());
      gst_pad_push_event (self->srcpad, gst_event_new_stream_start (sid));
      g_free (sid);

      self->send_stream_start = FALSE;
    }

    /** @todo get config from collect-pads and set framerate */
    caps = gst_caps_from_string (GST_TENSORS_FLEX_CAP_DEFAULT);
    gst_caps_set_simple (caps, "framerate", GST_TYPE_FRACTION, 0, 1, NULL);

    gst_pad_set_caps (self->srcpad, caps);
    gst_caps_unref (caps);

    gst_segment_init (&segment, GST_FORMAT_TIME);
    gst_pad_push_event (self->srcpad, gst_event_new_segment (&segment));
  }

  return GST_FLOW_OK;
}

/**
 * @brief Chain function called when the buffer is available on all of the collect pads.
 */
static GstFlowReturn
gst_tensor_crop_collected (GstCollectPads * pads, gpointer user_data)
{
  GstTensorCrop *self;
  GstBuffer *raw_buffer, *info_buffer, *result;
  GSList *walk;
  GstFlowReturn ret;

  self = GST_TENSOR_CROP (user_data);
  raw_buffer = info_buffer = result = NULL;

  ret = gst_tensor_crop_negotiate (self);
  if (ret != GST_FLOW_OK)
    return ret;

  for (walk = pads->data; walk; walk = g_slist_next (walk)) {
    GstCollectData *data;

    data = (GstCollectData *) walk->data;

    /**
     * @todo add timestampe policy (base on raw data buffer)
     * The case when raw and info have different timestamp
     * - one possible option: add property latency to allow diff (-1 means no sync, positive value to wait for other data)
     */
    if (data->pad == self->sinkpad_raw) {
      raw_buffer = gst_collect_pads_pop (pads, data);
    } else if (data->pad == self->sinkpad_info) {
      info_buffer = gst_collect_pads_pop (pads, data);
    }
  }

  /**
   * @todo crop incoming buffer
   * 1. check multi tensor (tensor + crop-info)
   * 2. parse crop-info (defined struct, flexible)
   * 3. crop incoming tensor with crop-info
   * 4. finally push flex tensor
   */
  result = gst_buffer_ref (raw_buffer);

  if (raw_buffer)
    gst_buffer_unref (raw_buffer);
  if (info_buffer)
    gst_buffer_unref (info_buffer);

  if (result)
    ret = gst_pad_push (self->srcpad, result);

  return ret;
}
