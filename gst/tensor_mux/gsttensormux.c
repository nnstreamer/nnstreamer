/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
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
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * @file	gsttensormux.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to mux tensors (as a filter for other general neural network filters)
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
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
  PROP_TIMESTAMP_OFFSET,
  PROP_SILENT
};

#define DEFAULT_TIMESTAMP_OFFSET -1

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

#define gst_tensor_mux_parent_class parent_class

static GstPad *gst_tensor_mux_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps);
static void gst_tensor_mux_release_pad (GstElement * element, GstPad * pad);
static GstFlowReturn
gst_tensor_mux_chain (GstPad * pad, GstObject * parent, GstBuffer * buffer);
static gboolean
gst_tensor_mux_setcaps (GstPad * pad, GstTensorMux * tensor_mux,
    GstCaps * caps);
static gboolean gst_tensor_mux_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstStateChangeReturn gst_tensor_mux_change_state (GstElement * element,
    GstStateChange transition);
static void gst_tensor_mux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_mux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_mux_dispose (GObject * object);

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

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gobject_class->get_property = gst_tensor_mux_get_property;
  gobject_class->set_property = gst_tensor_mux_set_property;
  gobject_class->dispose = gst_tensor_mux_dispose;

  g_object_class_install_property (gobject_class,
      PROP_TIMESTAMP_OFFSET, g_param_spec_int ("timestamp-offset",
          "Timestamp Offset",
          "Offset to add to all outgoing timestamps (-1 = random)", -1,
          G_MAXINT, DEFAULT_TIMESTAMP_OFFSET,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_request_new_pad);
  gstelement_class->release_pad =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_release_pad);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_mux_change_state);

  gst_element_class_set_details_simple (gstelement_class,
      "tensormux",
      "Mux multiple tensor stream",
      "Merge multiple tensor stream to tensors stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

}

/**
 * @brief dispose tensor mux / tensor mux pad object
 * @param object GstTensorMux*
 */
static void
gst_tensor_mux_dispose (GObject * object)
{
  GstTensorMux *tensor_mux = GST_TENSOR_MUX (object);
  GList *item;

  g_clear_object (&tensor_mux->last_pad);

restart:
  for (item = GST_ELEMENT_PADS (object); item; item = g_list_next (item)) {
    GstPad *pad = GST_PAD (item->data);
    if (GST_PAD_IS_SINK (pad)) {
      gst_element_release_request_pad (GST_ELEMENT (object), pad);
      goto restart;
    }
  }

  G_OBJECT_CLASS (gst_tensor_mux_parent_class)->dispose (object);
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
  gst_pad_use_fixed_caps (tensor_mux->srcpad);
  gst_element_add_pad (GST_ELEMENT (tensor_mux), tensor_mux->srcpad);

  tensor_mux->num_tensors = 0;
  tensor_mux->byte_count = 0;
  tensor_mux->silent = FALSE;
  tensor_mux->first = TRUE;
  tensor_mux->rank = 0;
  tensor_mux->dimensions = g_string_new (NULL);
  tensor_mux->types = g_string_new (NULL);
  tensor_mux->outbuffer = gst_buffer_new ();
}

/**
 * @brief Setup Sink Pad for tensor mux
 * Initialize pad private data & set functions for pad
 * @param tensor_mux GstTensorMux Pointer
 * @param sinkpad Sink pad
 */
static void
gst_tensor_mux_setup_sinkpad (GstTensorMux * tensor_mux, GstPad * sinkpad)
{
  GstTensorMuxPadPrivate *padpriv = g_slice_new0 (GstTensorMuxPadPrivate);
  padpriv->done = FALSE;
  gst_pad_set_chain_function (sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_mux_chain));
  gst_pad_set_event_function (sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_mux_sink_event));

  gst_pad_set_element_private (sinkpad, padpriv);

  gst_pad_set_active (sinkpad, TRUE);
  gst_element_add_pad (GST_ELEMENT (tensor_mux), sinkpad);
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

  g_return_val_if_fail (templ != NULL, NULL);
  g_return_val_if_fail (GST_IS_TENSOR_MUX (element), NULL);

  tensor_mux = GST_TENSOR_MUX (element);

  if (templ->direction != GST_PAD_SINK) {
    GST_WARNING_OBJECT (tensor_mux, "request pad that is not a SINK pad");
    return NULL;
  }

  newpad = gst_pad_new_from_template (templ, req_name);

  if (newpad)
    gst_tensor_mux_setup_sinkpad (tensor_mux, newpad);
  else
    GST_WARNING_OBJECT (tensor_mux, "failed to create request pad");

  return newpad;
}

/**
 * @brief release request pad (gst element vmethod)
 */
static void
gst_tensor_mux_release_pad (GstElement * element, GstPad * pad)
{
  GstTensorMuxPadPrivate *padpriv;
  GstTensorMux *tensor_mux = GST_TENSOR_MUX_CAST (element);

  GST_DEBUG_OBJECT (tensor_mux, "releaseing pad %s:%s",
      GST_DEBUG_PAD_NAME (pad));

  gst_pad_set_active (pad, FALSE);

  GST_OBJECT_LOCK (element);
  padpriv = gst_pad_get_element_private (pad);
  gst_pad_set_element_private (pad, NULL);
  GST_OBJECT_UNLOCK (element);
  gst_element_remove_pad (GST_ELEMENT_CAST (tensor_mux), pad);
  if (padpriv) {
    g_slice_free (GstTensorMuxPadPrivate, padpriv);
  }
}

/**
 * @brief resend events for sticky events
 */
static gboolean
resend_events (GstPad * pad, GstEvent ** event, gpointer user_data)
{
  GstTensorMux *tensor_mux = user_data;

  if (GST_EVENT_TYPE (*event) == GST_EVENT_CAPS) {
    GstCaps *caps;

    gst_event_parse_caps (*event, &caps);
    gst_tensor_mux_setcaps (pad, tensor_mux, caps);
  } else if (GST_EVENT_TYPE (*event) == GST_EVENT_SEGMENT) {
    GstSegment new_segment;
    gst_segment_init (&new_segment, GST_FORMAT_TIME);
    gst_pad_push_event (tensor_mux->srcpad,
        gst_event_new_segment (&new_segment));
  } else {
    gst_pad_push_event (tensor_mux->srcpad, gst_event_ref (*event));
  }

  return TRUE;
}

/**
 * @brief chain function (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_mux_chain (GstPad * pad, GstObject * parent, GstBuffer * buffer)
{
  GstTensorMux *tensor_mux;
  GstFlowReturn ret;
  GstTensorMuxPadPrivate *padpriv;
  gboolean changed = FALSE;
  tensor_mux = GST_TENSOR_MUX (parent);

  GST_DEBUG_OBJECT (pad, "recevied %" GST_PTR_FORMAT, buffer);

  if (gst_pad_check_reconfigure (tensor_mux->srcpad)) {
    GstCaps *current_caps = gst_pad_get_current_caps (pad);
    if (!gst_tensor_mux_setcaps (pad, tensor_mux, current_caps)) {
      gst_pad_mark_reconfigure (tensor_mux->srcpad);
      if (GST_PAD_IS_FLUSHING (tensor_mux->srcpad))
        ret = GST_FLOW_FLUSHING;
      else
        ret = GST_FLOW_NOT_NEGOTIATED;
      goto out;
    }
    gst_caps_unref (current_caps);
  }

  GST_OBJECT_LOCK (tensor_mux);
  padpriv = gst_pad_get_element_private (pad);

  if (!padpriv) {
    GST_OBJECT_UNLOCK (tensor_mux);
    gst_buffer_unref (buffer);
    return GST_FLOW_NOT_LINKED;
  }

  buffer = gst_buffer_make_writable (buffer);


  /* @TODO tensors buffer & data with tensor */

  if (pad != tensor_mux->last_pad) {
    changed = TRUE;
    g_clear_object (&tensor_mux->last_pad);
    tensor_mux->last_pad = g_object_ref (pad);
  }

  if (GST_BUFFER_DURATION_IS_VALID (buffer) && GST_BUFFER_PTS_IS_VALID (buffer))
    tensor_mux->last_stop =
        GST_BUFFER_PTS (buffer) + GST_BUFFER_DURATION (buffer);
  else
    tensor_mux->last_stop = GST_CLOCK_TIME_NONE;

  GST_OBJECT_UNLOCK (tensor_mux);

  if (changed)
    gst_pad_sticky_events_foreach (pad, resend_events, tensor_mux);

  gst_buffer_unref (buffer);
  gst_buffer_ref (tensor_mux->outbuffer);
  ret = gst_pad_push (tensor_mux->srcpad, tensor_mux->outbuffer);
out:
  return ret;
}


/**
 * @brief set caps (gst element vmethod)
 */
static gboolean
gst_tensor_mux_setcaps (GstPad * pad, GstTensorMux * tensor_mux, GstCaps * caps)
{
  GstStructure *structure;
  gboolean ret = FALSE;
  GstTensorMuxPadPrivate *padpriv;
  GstCaps *peercaps, *src_caps;
  gint dim;
  const gchar *t;

  GST_DEBUG_OBJECT (tensor_mux, "setcaps for pad %s:%s",
      GST_DEBUG_PAD_NAME (pad));

  padpriv = gst_pad_get_element_private (pad);
  if (padpriv->done)
    return TRUE;
  if (!gst_caps_is_fixed (caps))
    return FALSE;

  peercaps = gst_pad_peer_query_caps (tensor_mux->srcpad, NULL);
  if (peercaps) {
    GstCaps *tcaps, *othercaps;
    tcaps = gst_pad_get_pad_template_caps (pad);
    othercaps =
        gst_caps_intersect_full (peercaps, tcaps, GST_CAPS_INTERSECT_FIRST);

    if (gst_caps_get_size (othercaps) > 0) {
      structure = gst_caps_get_structure (othercaps, 0);
      GST_OBJECT_LOCK (tensor_mux);
      tensor_mux->first = FALSE;
      GST_OBJECT_UNLOCK (tensor_mux);
    }

    gst_caps_unref (othercaps);
    gst_caps_unref (peercaps);
    gst_caps_unref (tcaps);
  }


  structure = gst_caps_get_structure (caps, 0);
  if (!structure)
    return FALSE;

  GST_OBJECT_LOCK (tensor_mux);
  tensor_mux->num_tensors++;
  gst_structure_get_int (structure, "rank", &tensor_mux->rank);
  gst_structure_get_int (structure, "dim1", &dim);
  if (tensor_mux->dimensions->len != 0)
    tensor_mux->dimensions = g_string_append (tensor_mux->dimensions, ",");
  tensor_mux->dimensions =
      g_string_append (tensor_mux->dimensions, g_strdup_printf ("%d", dim));
  tensor_mux->dimensions = g_string_append (tensor_mux->dimensions, ":");
  gst_structure_get_int (structure, "dim2", &dim);
  tensor_mux->dimensions =
      g_string_append (tensor_mux->dimensions, g_strdup_printf ("%d", dim));
  tensor_mux->dimensions = g_string_append (tensor_mux->dimensions, ":");
  gst_structure_get_int (structure, "dim3", &dim);
  tensor_mux->dimensions =
      g_string_append (tensor_mux->dimensions, g_strdup_printf ("%d", dim));
  tensor_mux->dimensions = g_string_append (tensor_mux->dimensions, ":");
  gst_structure_get_int (structure, "dim4", &dim);
  tensor_mux->dimensions =
      g_string_append (tensor_mux->dimensions, g_strdup_printf ("%d", dim));

  if (tensor_mux->types->len != 0)
    tensor_mux->types = g_string_append (tensor_mux->types, ",");
  t = gst_structure_get_string (structure, "type");
  tensor_mux->types = g_string_append (tensor_mux->types, t);


  gst_structure_get_fraction (structure, "framerate",
      &tensor_mux->framerate_numerator, &tensor_mux->framerate_denominator);

  padpriv = gst_pad_get_element_private (pad);
  if (padpriv
      && gst_structure_get_uint (structure, "timestamps-offset",
          &padpriv->timestamp_offset)) {
    padpriv->have_timestamp_offset = TRUE;
  }

  padpriv->done = TRUE;
  src_caps = gst_caps_new_simple ("other/tensors",
      "rank", G_TYPE_INT, tensor_mux->rank,
      "num_tensors", G_TYPE_INT, tensor_mux->num_tensors,
      "types", G_TYPE_STRING, tensor_mux->types->str,
      "framerate", GST_TYPE_FRACTION, tensor_mux->framerate_numerator,
      tensor_mux->framerate_denominator, "dimensions", G_TYPE_STRING,
      tensor_mux->dimensions->str, NULL);

  GST_OBJECT_UNLOCK (tensor_mux);

  if (tensor_mux->send_stream_start) {
    gchar s_id[32];

    g_snprintf (s_id, sizeof (s_id), "interleave-%08x", g_random_int ());
    gst_pad_push_event (tensor_mux->srcpad, gst_event_new_stream_start (s_id));

    tensor_mux->send_stream_start = FALSE;
  }

  GST_DEBUG_OBJECT (tensor_mux,
      "setting caps %" GST_PTR_FORMAT " on src pad..", src_caps);

  ret = gst_pad_set_caps (tensor_mux->srcpad, src_caps);
  gst_caps_unref (src_caps);

  return ret;
}

/**
 * @brief event function (gst element vmethod)
 */
static gboolean
gst_tensor_mux_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorMux *mux = GST_TENSOR_MUX (parent);
  gboolean is_pad;
  gboolean ret = TRUE;

  GST_OBJECT_LOCK (mux);
  is_pad = (pad == mux->last_pad);
  GST_OBJECT_UNLOCK (mux);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      GST_LOG_OBJECT (pad, "Received caps-event with caps: %"
          GST_PTR_FORMAT, caps);
      ret = gst_tensor_mux_setcaps (pad, mux, caps);
      gst_event_unref (event);
      return ret;
    }
      break;
    case GST_EVENT_FLUSH_STOP:
    {
      GST_OBJECT_LOCK (mux);
      mux->last_stop = GST_CLOCK_TIME_NONE;
      GST_OBJECT_UNLOCK (mux);
      break;
    }
    case GST_EVENT_SEGMENT:
    {
      GstTensorMuxPadPrivate *padpriv;
      GST_OBJECT_LOCK (mux);
      padpriv = gst_pad_get_element_private (pad);

      if (padpriv) {
        gst_event_copy_segment (event, &padpriv->segment);
      }
      GST_OBJECT_UNLOCK (mux);

      if (is_pad) {
        GstSegment new_segment;
        gst_segment_init (&new_segment, GST_FORMAT_TIME);
        gst_event_unref (event);
        event = gst_event_new_segment (&new_segment);
      }
      break;
    }
    default:
      break;
  }

  if (is_pad) {
    return gst_pad_push_event (mux->srcpad, event);
  } else {
    gst_event_unref (event);
    return ret;
  }
}

/**
 * @brief Ready --> Pasuse State Change
 */
static void
gst_tensor_mux_ready_to_paused (GstTensorMux * tensor_mux)
{
  GST_OBJECT_LOCK (tensor_mux);
  g_clear_object (&tensor_mux->last_pad);
  tensor_mux->send_stream_start = TRUE;
  GST_OBJECT_UNLOCK (tensor_mux);
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
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (gst_tensor_mux_parent_class)->change_state (element,
      transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      g_clear_object (&tensor_mux->last_pad);
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
static gboolean
gst_tensor_mux_plugin_init (GstPlugin * tensormux)
{
  /* debug category for fltering log messages
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
