/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
 *
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
 * @file	gsttensormux.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to mux tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensormux
 *
 * A Muxer that merge tensor stream to tensors stream for NN frameworks.
 * The output is always in the format of other/tensors
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m \
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 \
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 \
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 \
 * tensor_mux name=mux ! fakesink
 * ]|
 *
 * |[
 * gst-launch -v -m \
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_0 \
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_1 \
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_2 \
 * tensor_mux name=mux ! filesink location=mux.log
 * ]|
 * </refsect2 >
 *
 */


#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <gst/gst.h>
#include <glib.h>

#include "gsttensormux.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_mux_debug);
#define GST_CAT_DEFAULT gst_tensor_mux_debug

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!tensor_mux->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (tensor_mux, __VA_ARGS__); \
    } \
  } while (0)

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_SYNC_MODE,
  PROP_SYNC_OPTION,
};

/**
 * @brief Default caps string for sink pad.
 */
#define CAPS_STRING_SINK GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT

/**
 * @brief the capabilities of the inputs and outputs.
 * describe the real formats here.
 */
static GstStaticPadTemplate src_templ = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

static GstStaticPadTemplate sink_templ = GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (CAPS_STRING_SINK)
    );

static gboolean gst_tensor_mux_src_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstPad *gst_tensor_mux_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps);
static GstStateChangeReturn gst_tensor_mux_change_state (GstElement * element,
    GstStateChange transition);
static gboolean gst_tensor_mux_sink_event (GstCollectPads * pads,
    GstCollectData * data, GstEvent * event, GstTensorMux * tensor_mux);
static GstFlowReturn gst_tensor_mux_collected (GstCollectPads * pads,
    GstTensorMux * tesnor_mux);

static void gst_tensor_mux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_mux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_mux_finalize (GObject * object);

#define gst_tensor_mux_parent_class parent_class
G_DEFINE_TYPE (GstTensorMux, gst_tensor_mux, GST_TYPE_ELEMENT);

/**
 * @brief initialize the tensor_mux's class
 */
static void
gst_tensor_mux_class_init (GstTensorMuxClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_mux_debug, "tensor_mux", 0,
      "Element to merge tensor stream to tensors stream");

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->finalize = gst_tensor_mux_finalize;
  gobject_class->get_property = gst_tensor_mux_get_property;
  gobject_class->set_property = gst_tensor_mux_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          TRUE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SYNC_MODE,
      g_param_spec_string ("sync_mode", "Sync_Mode",
          "Time synchronization mode?", "", G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SYNC_OPTION,
      g_param_spec_string ("sync_option", "Sync_Option",
          "Option for the time synchronization mode ?", "", G_PARAM_READWRITE));

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_request_new_pad);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorMux",
      "Muxer/Tensor",
      "Merge multiple tensor stream to tensors stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_mux_init (GstTensorMux * tensor_mux)
{
  GstElementClass *klass = GST_ELEMENT_GET_CLASS (tensor_mux);

  tensor_mux->srcpad =
      gst_pad_new_from_template (gst_element_class_get_pad_template (klass,
          "src"), "src");
  gst_pad_set_event_function (tensor_mux->srcpad, gst_tensor_mux_src_event);

  gst_element_add_pad (GST_ELEMENT (tensor_mux), tensor_mux->srcpad);

  tensor_mux->collect = gst_collect_pads_new ();
  gst_collect_pads_set_event_function (tensor_mux->collect,
      (GstCollectPadsEventFunction)
      GST_DEBUG_FUNCPTR (gst_tensor_mux_sink_event), tensor_mux);
  gst_collect_pads_set_function (tensor_mux->collect,
      (GstCollectPadsFunction) GST_DEBUG_FUNCPTR (gst_tensor_mux_collected),
      tensor_mux);

  tensor_mux->silent = TRUE;
  tensor_mux->sync.mode = SYNC_SLOWEST;
  tensor_mux->sync.option = NULL;
  tensor_mux->current_time = 0;
  tensor_mux->need_set_time = TRUE;
  gst_tensors_config_init (&tensor_mux->tensors_config);
}

/**
 * @brief finalize vmethod
 */
static void
gst_tensor_mux_finalize (GObject * object)
{
  GstTensorMux *tensor_mux;

  tensor_mux = GST_TENSOR_MUX (object);

  if (tensor_mux->collect) {
    gst_object_unref (tensor_mux->collect);
    tensor_mux->collect = NULL;
  }

  if (tensor_mux->sync.option) {
    g_free (tensor_mux->sync.option);
    tensor_mux->sync.option = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief making new request pad (gst element vmethod)
 */
static GstPad *
gst_tensor_mux_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * req_name, const GstCaps * caps)
{
  GstPad *newpad;
  GSList *walk = NULL;
  GstTensorMux *tensor_mux;
  gchar *name;

  g_return_val_if_fail (templ != NULL, NULL);
  g_return_val_if_fail (GST_IS_TENSOR_MUX (element), NULL);

  tensor_mux = GST_TENSOR_MUX (element);
  walk = tensor_mux->collect->data;

  name = g_strdup_printf ("sink_%u", g_slist_length (walk));
  newpad = gst_pad_new_from_template (templ, name);
  g_free (name);

  if (newpad) {
    GstTensorCollectPadData *tensormuxpad;

    tensormuxpad = (GstTensorCollectPadData *)
        gst_collect_pads_add_pad (tensor_mux->collect, newpad,
        sizeof (GstTensorCollectPadData), NULL, TRUE);

    tensormuxpad->pad = newpad;
    gst_pad_set_element_private (newpad, tensormuxpad);
    gst_element_add_pad (element, newpad);
  } else {
    GST_WARNING_OBJECT (tensor_mux, "failed to create request pad");
  }
  return newpad;
}

/**
 * @brief src event vmethod
 */
static gboolean
gst_tensor_mux_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
      gst_event_unref (event);
      return FALSE;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief sink event vmethod
 */
static gboolean
gst_tensor_mux_sink_event (GstCollectPads * pads, GstCollectData * data,
    GstEvent * event, GstTensorMux * tensor_mux)
{
  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_STOP:
      tensor_mux->need_segment = TRUE;
      break;
    default:
      break;
  }

  return gst_collect_pads_event_default (pads, data, event, FALSE);
}

/**
 * @brief Looping to generete outbut buffer for srcpad
 * @param tensor_mux tensor muxer
 * @param tensors_buf output buffer for srcpad
 * @param is_eos boolean EOS ( End of Stream )
 * @return TRUE to push buffer to src pad
 */
static gboolean
gst_tensor_mux_collect_buffer (GstTensorMux * tensor_mux,
    GstBuffer * tensors_buf, gboolean * is_eos)
{
  if (tensor_mux->need_set_time) {
    if (gst_tensor_time_sync_get_current_time (tensor_mux->collect,
            &tensor_mux->sync, &tensor_mux->current_time)) {
      /* end-of-stream */
      *is_eos = TRUE;
      return FALSE;
    }

    tensor_mux->need_set_time = FALSE;
    silent_debug ("Current Time : %" GST_TIME_FORMAT,
        GST_TIME_ARGS (tensor_mux->current_time));
  }

  return gst_tensor_time_sync_buffer_from_collectpad (tensor_mux->collect,
      &tensor_mux->sync, tensor_mux->current_time, tensors_buf,
      &tensor_mux->tensors_config, is_eos);
}

/**
 * @brief Set src pad caps if src pad is not negotiated.
 */
static gboolean
gst_tensor_mux_set_src_caps (GstTensorMux * tensor_mux)
{
  if (!tensor_mux->negotiated) {
    GstCaps *caps;

    if (gst_tensors_config_validate (&tensor_mux->tensors_config)) {
      caps = gst_tensors_caps_from_config (&tensor_mux->tensors_config);

      if (gst_pad_set_caps (tensor_mux->srcpad, caps)) {
        tensor_mux->negotiated = TRUE;
      }

      gst_caps_unref (caps);
    }
  }

  if (!tensor_mux->negotiated) {
    GST_WARNING_OBJECT (tensor_mux, "failed to set caps");
    GST_ELEMENT_ERROR (tensor_mux, CORE, NEGOTIATION, (NULL), (NULL));
  }

  return tensor_mux->negotiated;
}

/**
 * @brief Create a new segment event if necessary.
 */
static void
gst_tensor_mux_send_segment_event (GstTensorMux * tensor_mux,
    GstClockTime pts, GstClockTime dts)
{
  if (tensor_mux->need_segment) {
    GstSegment segment;
    GstClockTime time = 0;

    if (GST_CLOCK_TIME_IS_VALID (dts)) {
      time = dts;
    } else if (GST_CLOCK_TIME_IS_VALID (pts)) {
      time = pts;
    }

    gst_segment_init (&segment, GST_FORMAT_TIME);
    segment.start = time;
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_segment (&segment));
    tensor_mux->need_segment = FALSE;
  }
}

/**
 * @brief Gst Collect Pads Function which is called once collect pads done.
 * @param pads GstCollectPads
 * @param tensor_mux Muxer
 * @return GstFlowReturn
 */
static GstFlowReturn
gst_tensor_mux_collected (GstCollectPads * pads, GstTensorMux * tensor_mux)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *tensors_buf;
  gboolean isEOS = FALSE;

  GST_DEBUG_OBJECT (tensor_mux, " all pads are collected ");

  if (tensor_mux->need_stream_start) {
    gchar s_id[32];
    g_snprintf (s_id, sizeof (s_id), " tensormux - %08x ", g_random_int ());
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_stream_start (s_id));
    tensor_mux->need_stream_start = FALSE;
  }

  if ((tensors_buf = gst_buffer_new ()) == NULL) {
    ml_logf ("gst_buffer_new() returns NULL. Out of memory?\n");
    return GST_FLOW_ERROR;
  }

  if (!gst_tensor_mux_collect_buffer (tensor_mux, tensors_buf, &isEOS)) {
    if (isEOS) {
      gst_pad_push_event (tensor_mux->srcpad, gst_event_new_eos ());
      ret = GST_FLOW_EOS;
    }

    gst_buffer_unref (tensors_buf);
    return ret;
  }

  if (!gst_tensor_mux_set_src_caps (tensor_mux)) {
    gst_buffer_unref (tensors_buf);
    return GST_FLOW_NOT_NEGOTIATED;
  }

  gst_tensor_mux_send_segment_event (tensor_mux, GST_BUFFER_PTS (tensors_buf),
      GST_BUFFER_DTS (tensors_buf));

  ret = gst_pad_push (tensor_mux->srcpad, tensors_buf);
  tensor_mux->need_set_time = TRUE;

  if (ret != GST_FLOW_OK) {
    GST_WARNING_OBJECT (tensor_mux, "pushed outbuf, result = %s",
        gst_flow_get_name (ret));
  }

  return ret;
}

/**
 * @brief Ready --> Pasuse State Change
 */
static void
gst_tensor_mux_ready_to_paused (GstTensorMux * tensor_mux)
{
  tensor_mux->need_stream_start = TRUE;
  tensor_mux->need_segment = TRUE;
  tensor_mux->negotiated = FALSE;
  gst_collect_pads_start (tensor_mux->collect);
}

/**
 * @brief change state (gst element vmethod)
 */
static GstStateChangeReturn
gst_tensor_mux_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorMux *tensor_mux;
  GstStateChangeReturn ret;
  tensor_mux = GST_TENSOR_MUX (element);
  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_tensor_mux_ready_to_paused (tensor_mux);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_collect_pads_stop (tensor_mux->collect);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return ret;
  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_mux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorMux *tensor_mux = GST_TENSOR_MUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      tensor_mux->silent = g_value_get_boolean (value);
      break;
    case PROP_SYNC_MODE:
      tensor_mux->sync.mode =
          gst_tensor_time_sync_get_mode (g_value_get_string (value));
      if (tensor_mux->sync.mode == SYNC_END) {
        tensor_mux->sync.mode = SYNC_SLOWEST;
      }
      silent_debug ("Mode = %d(%s)\n", tensor_mux->sync.mode,
          gst_tensor_time_sync_get_mode_string (tensor_mux->sync.mode));
      gst_tensor_time_sync_set_option_data (&tensor_mux->sync);
      break;
    case PROP_SYNC_OPTION:
      tensor_mux->sync.option = g_value_dup_string (value);
      silent_debug ("Option = %s\n", tensor_mux->sync.option);
      gst_tensor_time_sync_set_option_data (&tensor_mux->sync);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_mux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorMux *tensor_mux = GST_TENSOR_MUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, tensor_mux->silent);
      break;
    case PROP_SYNC_MODE:
      g_value_set_string (value,
          gst_tensor_time_sync_get_mode_string (tensor_mux->sync.mode));
      break;
    case PROP_SYNC_OPTION:
      g_value_set_string (value, tensor_mux->sync.option);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}
