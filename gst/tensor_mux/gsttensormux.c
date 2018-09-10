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
 * @bug         No known bugs
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
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
 * gst-launch -v -m tensormux name=mux ! fakesink
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2
 * ]|
 *
 * |[
 * gst-launch -v -m tensormux name=mux ! filesink location=mux.log
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
#include <tensor_meta.h>

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

static void gst_tensor_mux_finalize (GObject * object);

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
          FALSE, G_PARAM_READWRITE));

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_request_new_pad);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "tensormux",
      "Mux multiple tensor stream",
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

  tensor_mux->num_tensors = 0;
  tensor_mux->silent = FALSE;
  tensor_mux->rank = 0;
}

/**
 * @brief finalize vmethod
 */
static void
gst_tensor_mux_finalize (GObject * object)
{
  GstTensorMux *tensor_mux;
  tensor_mux = GST_TENSOR_MUX (object);

  if (tensor_mux->collect)
    gst_object_unref (tensor_mux->collect);
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

  name = g_strdup_printf ("sink_%u", tensor_mux->num_tensors);
  newpad = gst_pad_new_from_template (templ, name);
  g_free (name);

  if (newpad) {
    GstTensorMuxPadData *tensormuxpad;
    tensormuxpad =
        (GstTensorMuxPadData *) gst_collect_pads_add_pad (tensor_mux->collect,
        newpad, sizeof (GstTensorMuxPadData), NULL, TRUE);
    tensormuxpad->pad = newpad;
    gst_pad_set_element_private (newpad, tensormuxpad);
    tensor_mux->num_tensors++;
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
 * @param old previous mux pad data
 * @param new current mux pad data
 * @return if > 0, new is earlier than old
 */
static gint
gst_tensor_mux_compare_pads (GstTensorMux * tensor_mux,
    GstTensorMuxPadData * old, GstTensorMuxPadData * new)
{
  guint64 oldtime, newtime;
  if (old == NULL || old->buffer == NULL)
    return 1;
  if (new == NULL || new->buffer == NULL)
    return -1;
  if (GST_CLOCK_TIME_IS_VALID (old->dts_timestamp) &&
      GST_CLOCK_TIME_IS_VALID (new->dts_timestamp)) {
    oldtime = old->dts_timestamp;
    newtime = new->dts_timestamp;
  } else {
    oldtime = old->pts_timestamp;
    newtime = new->pts_timestamp;
  }

  if (oldtime == GST_CLOCK_TIME_NONE)
    return -1;
  if (newtime == GST_CLOCK_TIME_NONE)
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
 * @param dimensions collected dimensions as string
 * @param types collected types as string
 * @param pts_time earliest pts time (present timestamp)
 * @param dts_time earliest dts time (decoding timestamp)
 * @return isEOS boolean EOS ( End of Stream )
 */
gboolean
gst_tensor_mux_collect_buffer (GstTensorMux * tensor_mux,
    GstBuffer * tensors_buf, GString * dimensions, GString * types,
    GstClockTime * pts_time, GstClockTime * dts_time)
{
  GSList *walk = NULL;
  GstTensorMuxPadData *bestpad = NULL;

  tensor_dim dim;
  tensor_type type;
  GstMemory *mem;
  GstTensor_Filter_CheckStatus status;
  gboolean isEOS = FALSE;
  gint i;

  gint old_numerator = 2147483647, old_denominator = 2147483647;
  gint new_numerator, new_denominator;
  gint counting = 0;
  tensor_mux->rank = NNS_TENSOR_RANK_LIMIT;

  walk = tensor_mux->collect->data;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorMuxPadData *pad = (GstTensorMuxPadData *) data;

    GstCaps *caps = gst_pad_get_current_caps (pad->pad);
    status =
        get_tensor_from_padcap (caps, dim, &type, &new_numerator,
        &new_denominator);
    g_assert ((status & _TFC_ALL) == _TFC_ALL);

    if (new_denominator < old_denominator)
      old_denominator = new_denominator;
    if (new_numerator < old_numerator)
      old_numerator = new_numerator;

    if (dimensions->len != 0)
      dimensions = g_string_append (dimensions, ",");
    for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
      dimensions = g_string_append (dimensions, g_strdup_printf ("%d", dim[i]));
      if (i < NNS_TENSOR_RANK_LIMIT - 1)
        dimensions = g_string_append (dimensions, ":");
    }
    if (types->len != 0)
      types = g_string_append (types, ",");
    types = g_string_append (types, tensor_element_typename[type]);
    gst_caps_unref (caps);
    walk = g_slist_next (walk);
    GstBuffer *buf = NULL;
    buf = gst_collect_pads_pop (tensor_mux->collect, data);
    if (buf && GST_BUFFER_PTS_IS_VALID (buf)) {
      pad->pts_timestamp =
          gst_segment_to_running_time (&data->segment, GST_FORMAT_TIME,
          GST_BUFFER_PTS (buf));
    } else {
      pad->pts_timestamp = GST_CLOCK_TIME_NONE;
    }
    if (buf && GST_BUFFER_DTS_IS_VALID (buf)) {
      pad->dts_timestamp =
          gst_segment_to_running_time (&data->segment, GST_FORMAT_TIME,
          GST_BUFFER_DTS (buf));
    } else {
      pad->dts_timestamp = GST_CLOCK_TIME_NONE;
    }

    pad->buffer = buf;
    if (GST_IS_BUFFER (buf)) {
      gst_buffer_unref (pad->buffer);
      mem = gst_buffer_get_memory (buf, 0);
      gst_memory_ref (mem);
      if (!gst_append_tensor (tensors_buf, mem, dim, type, counting))
        return FALSE;
      if (pad->buffer != NULL)
        if (gst_tensor_mux_compare_pads (tensor_mux, bestpad, pad) > 0) {
          bestpad = pad;
          *pts_time = bestpad->pts_timestamp;
          *dts_time = bestpad->dts_timestamp;
        }
    } else {
      isEOS = TRUE;
    }
    counting++;
  }

  tensor_mux->framerate_denominator = old_denominator;
  tensor_mux->framerate_numerator = old_numerator;

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
  GString *dimensions = g_string_new (NULL);
  GString *types = g_string_new (NULL);
  GstClockTime pts_time, dts_time;
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
  gst_make_tensors (tensors_buf);
  isEOS =
      gst_tensor_mux_collect_buffer (tensor_mux, tensors_buf, dimensions,
      types, &pts_time, &dts_time);
  if (isEOS) {
    if (tensors_buf)
      gst_buffer_unref (tensors_buf);
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_eos ());
    ret = GST_FLOW_EOS;
    goto beach;
  }

  if (!tensor_mux->negotiated) {
    GstCaps *newcaps;
    newcaps =
        gst_caps_new_simple ("other/tensors",
        "num_tensors", G_TYPE_INT,
        tensor_mux->num_tensors, "types", G_TYPE_STRING,
        types->str, "framerate", GST_TYPE_FRACTION,
        tensor_mux->framerate_numerator, tensor_mux->framerate_denominator,
        "dimensions", G_TYPE_STRING, dimensions->str, NULL);
    if (!gst_pad_set_caps (tensor_mux->srcpad, newcaps)) {
      gst_caps_unref (newcaps);
      goto nego_error;
    }

    gst_caps_unref (newcaps);
    tensor_mux->negotiated = TRUE;
  }

  if (tensor_mux->need_segment) {
    GstSegment segment;
    if (dts_time != GST_CLOCK_TIME_NONE) {
      time = dts_time;
    } else if (pts_time != GST_CLOCK_TIME_NONE) {
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
  if (ret != GST_FLOW_OK)
    goto beach;
beach:
  return ret;
nego_error:
  {
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
 * PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "tensormux"
#endif

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
gboolean
gst_tensor_mux_plugin_init (GstPlugin * tensormux)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_mux' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_mux_debug, "tensormux", 0,
      "Tensor Muxer");
  return gst_element_register (tensormux, "tensormux",
      GST_RANK_NONE, GST_TYPE_TENSOR_MUX);
}

/**
 * @brief gstreamer looks for this structure to register tensormux
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensormux,
    "tensormux",
    gst_tensor_mux_plugin_init, VERSION, "LGPL", "GStreamer",
    "http://gstreamer.net/");
