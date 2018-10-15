/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
 *
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
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
 * @see		https://github.com/nnsuite/nnstreamer
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
 * gst-launch -v -m tensor_mux name=mux ! fakesink
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2
 * ]|
 *
 * |[
 * gst-launch -v -m tensor_mux name=mux ! filesink location=mux.log
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_0
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_1
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_2
 *
 * </refsect2>
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

enum
{
  PROP_0,
  PROP_SILENT
};

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
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static gboolean gst_tensor_mux_handle_src_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
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

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->finalize = gst_tensor_mux_finalize;
  gobject_class->get_property = gst_tensor_mux_get_property;
  gobject_class->set_property = gst_tensor_mux_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          TRUE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  gst_pad_set_event_function (tensor_mux->srcpad,
      gst_tensor_mux_handle_src_event);

  gst_element_add_pad (GST_ELEMENT (tensor_mux), tensor_mux->srcpad);

  tensor_mux->collect = gst_collect_pads_new ();
  gst_collect_pads_set_event_function (tensor_mux->collect,
      (GstCollectPadsEventFunction)
      GST_DEBUG_FUNCPTR (gst_tensor_mux_sink_event), tensor_mux);
  gst_collect_pads_set_function (tensor_mux->collect,
      (GstCollectPadsFunction) GST_DEBUG_FUNCPTR (gst_tensor_mux_collected),
      tensor_mux);

  tensor_mux->silent = TRUE;
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
  GstTensorMux *tensor_mux;
  gchar *name;

  g_return_val_if_fail (templ != NULL, NULL);
  g_return_val_if_fail (GST_IS_TENSOR_MUX (element), NULL);

  tensor_mux = GST_TENSOR_MUX (element);

  name =
      g_strdup_printf ("sink_%u", tensor_mux->tensors_config.info.num_tensors);
  newpad = gst_pad_new_from_template (templ, name);
  g_free (name);

  if (newpad) {
    GstTensorMuxPadData *tensormuxpad;
    tensormuxpad =
        (GstTensorMuxPadData *) gst_collect_pads_add_pad (tensor_mux->collect,
        newpad, sizeof (GstTensorMuxPadData), NULL, TRUE);
    tensormuxpad->pad = newpad;
    gst_pad_set_element_private (newpad, tensormuxpad);
    tensor_mux->tensors_config.info.num_tensors++;
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
gst_tensor_mux_handle_src_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstEventType type;
  type = event ? GST_EVENT_TYPE (event) : GST_EVENT_UNKNOWN;
  switch (type) {
    case GST_EVENT_SEEK:
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
  gboolean ret;
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_STOP:
    {
      tensor_mux->need_segment = TRUE;
      break;
    }
    default:
      break;
  }

  ret = gst_collect_pads_event_default (pads, data, event, FALSE);
  return ret;
}

/**
 * @brief Compare dts & pts time and find earliest
 * @param tensor_mux tensor muxer
 * @param old_data previous mux pad data
 * @param new_data current mux pad data
 * @return if > 0, new is earlier than old
 */
static gint
gst_tensor_mux_compare_pads (GstTensorMux * tensor_mux,
    GstTensorMuxPadData * old_data, GstTensorMuxPadData * new_data)
{
  guint64 oldtime, newtime;
  if (old_data == NULL)
    return 1;
  if (new_data == NULL)
    return -1;
  if (GST_CLOCK_TIME_IS_VALID (old_data->dts_timestamp) &&
      GST_CLOCK_TIME_IS_VALID (new_data->dts_timestamp)) {
    oldtime = old_data->dts_timestamp;
    newtime = new_data->dts_timestamp;
  } else {
    oldtime = old_data->pts_timestamp;
    newtime = new_data->pts_timestamp;
  }

  if (!GST_CLOCK_TIME_IS_VALID (oldtime))
    return -1;
  if (!GST_CLOCK_TIME_IS_VALID (newtime))
    return 1;

  if (newtime < oldtime)
    return 1;
  else if (newtime > oldtime)
    return -1;

  return 0;
}

/**
 * @brief Looping to generete outbut buffer for srcpad
 * @param tensor_mux tensor muxer
 * @param tensors_buf output buffer for srcpad
 * @param pts_time earliest pts time (present timestamp)
 * @param dts_time earliest dts time (decoding timestamp)
 * @return isEOS boolean EOS ( End of Stream )
 */
static gboolean
gst_tensor_mux_collect_buffer (GstTensorMux * tensor_mux,
    GstBuffer * tensors_buf, GstClockTime * pts_time, GstClockTime * dts_time)
{
  GSList *walk = NULL;
  GstTensorMuxPadData *bestpad = NULL;
  GstMemory *mem;
  gboolean isEOS = FALSE;
  gint old_numerator = G_MAXINT;
  gint old_denominator = G_MAXINT;
  gint counting = 0;
  GstTensorConfig config;

  walk = tensor_mux->collect->data;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorMuxPadData *pad = (GstTensorMuxPadData *) data;
    GstCaps *caps = gst_pad_get_current_caps (pad->pad);
    GstStructure *s = gst_caps_get_structure (caps, 0);

    gst_tensor_config_from_structure (&config, s);
    g_assert (gst_tensor_config_validate (&config));

    if (config.rate_d < old_denominator)
      old_denominator = config.rate_d;
    if (config.rate_n < old_numerator)
      old_numerator = config.rate_n;

    gst_caps_unref (caps);

    walk = g_slist_next (walk);

    GstBuffer *buf = NULL;
    buf = gst_collect_pads_pop (tensor_mux->collect, data);

    if (GST_IS_BUFFER (buf)) {
      if (GST_BUFFER_PTS_IS_VALID (buf)) {
        pad->pts_timestamp =
            gst_segment_to_running_time (&data->segment, GST_FORMAT_TIME,
            GST_BUFFER_PTS (buf));
      } else {
        pad->pts_timestamp = GST_CLOCK_TIME_NONE;
      }
      if (GST_BUFFER_DTS_IS_VALID (buf)) {
        pad->dts_timestamp =
            gst_segment_to_running_time (&data->segment, GST_FORMAT_TIME,
            GST_BUFFER_DTS (buf));
      } else {
        pad->dts_timestamp = GST_CLOCK_TIME_NONE;
      }

      mem = gst_buffer_get_memory (buf, 0);
      gst_buffer_append_memory (tensors_buf, mem);

      if (gst_tensor_mux_compare_pads (tensor_mux, bestpad, pad) > 0) {
        bestpad = pad;
        *pts_time = bestpad->pts_timestamp;
        *dts_time = bestpad->dts_timestamp;
      }

      gst_buffer_unref (buf);
    } else {
      isEOS = TRUE;
    }

    tensor_mux->tensors_config.info.info[counting] = config.info;
    counting++;
  }

  tensor_mux->tensors_config.rate_d = old_denominator;
  tensor_mux->tensors_config.rate_n = old_numerator;

  debug_print (!tensor_mux->silent, "pts %" GST_TIME_FORMAT,
      GST_TIME_ARGS (*pts_time));
  debug_print (!tensor_mux->silent, "dts %" GST_TIME_FORMAT,
      GST_TIME_ARGS (*dts_time));

  /* set timestamp */
  GST_BUFFER_PTS (tensors_buf) = *pts_time;
  GST_BUFFER_DTS (tensors_buf) = *dts_time;
  return isEOS;
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
  GstClockTime pts_time = GST_CLOCK_TIME_NONE;
  GstClockTime dts_time = GST_CLOCK_TIME_NONE;
  GstClockTime time = 0;
  gboolean isEOS = FALSE;
  GST_DEBUG_OBJECT (tensor_mux, " all pads are collected ");
  if (tensor_mux->need_stream_start) {
    gchar s_id[32];
    g_snprintf (s_id, sizeof (s_id), " tensormux - %08x ", g_random_int ());
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_stream_start (s_id));
    tensor_mux->need_stream_start = FALSE;
  }

  tensors_buf = gst_buffer_new ();
  g_assert (tensors_buf);

  isEOS =
      gst_tensor_mux_collect_buffer (tensor_mux, tensors_buf, &pts_time,
      &dts_time);

  if (isEOS) {
    gst_buffer_unref (tensors_buf);
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_eos ());
    ret = GST_FLOW_EOS;
    goto beach;
  }

  if (!tensor_mux->negotiated) {
    GstCaps *newcaps;

    g_assert (gst_tensors_config_validate (&tensor_mux->tensors_config));
    newcaps = gst_tensors_caps_from_config (&tensor_mux->tensors_config);

    if (!gst_pad_set_caps (tensor_mux->srcpad, newcaps)) {
      gst_caps_unref (newcaps);
      goto nego_error;
    }

    gst_caps_unref (newcaps);
    tensor_mux->negotiated = TRUE;
  }

  if (tensor_mux->need_segment) {
    GstSegment segment;

    if (GST_CLOCK_TIME_IS_VALID (dts_time)) {
      time = dts_time;
    } else if (GST_CLOCK_TIME_IS_VALID (pts_time)) {
      time = pts_time;
    } else {
      time = 0;
    }

    gst_segment_init (&segment, GST_FORMAT_TIME);
    segment.start = time;
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_segment (&segment));
    tensor_mux->need_segment = FALSE;
  }

  ret = gst_pad_push (tensor_mux->srcpad, tensors_buf);
  if (ret != GST_FLOW_OK) {
    GST_WARNING_OBJECT (tensor_mux, "pushed outbuf, result = %s",
        gst_flow_get_name (ret));
    /* fall-through, returns result */
  }
beach:
  return ret;
nego_error:
  {
    gst_buffer_unref (tensors_buf);
    GST_WARNING_OBJECT (tensor_mux, "failed to set caps");
    GST_ELEMENT_ERROR (tensor_mux, CORE, NEGOTIATION, (NULL), (NULL));
    return GST_FLOW_NOT_NEGOTIATED;
  }
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
  GstTensorMux *filter = GST_TENSOR_MUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
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
  GstTensorMux *filter = GST_TENSOR_MUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
NNSTREAMER_PLUGIN_INIT (tensor_mux)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_mux' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_mux_debug, "tensor_mux", 0,
      "Tensor Muxer");
  return gst_element_register (plugin, "tensor_mux",
      GST_RANK_NONE, GST_TYPE_TENSOR_MUX);
}

#ifndef SINGLE_BINARY
/**
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

/**
 * @brief gstreamer looks for this structure to register tensormux
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_mux,
    "tensor mux plugin",
    gst_tensor_mux_plugin_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer/");
#endif
