/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	gsttensordemux.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to demux tensors (as a filter for other general neural network filters)
 * @bug         No known bugs
 *
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensordemux
 *
 * A Deuxer that demux tensors stream to tensor stream for NN frameworks.
 * The outputs are always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * ]|
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

#include "gsttensordemux.h"
#include <tensor_meta.h>

GST_DEBUG_CATEGORY_STATIC (gst_tensor_demux_debug);
#define GST_CAT_DEFAULT gst_tensor_demux_debug

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_TENSORPICK
};

/**
 * @brief the capabilities of the inputs and outputs.
 * describe the real formats here.
 */
static GstStaticPadTemplate src_templ = GST_STATIC_PAD_TEMPLATE ("src_%u",
    GST_PAD_SRC,
    GST_PAD_SOMETIMES,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static GstStaticPadTemplate sink_templ = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT)
    );

static GstFlowReturn gst_tensor_demux_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);
static gboolean gst_tensor_demux_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstStateChangeReturn gst_tensor_demux_change_state (GstElement * element,
    GstStateChange transition);
static void gst_tensor_demux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_demux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_demux_dispose (GObject * object);
#define gst_tensor_demux_parent_class parent_class
G_DEFINE_TYPE (GstTensorDemux, gst_tensor_demux, GST_TYPE_ELEMENT);


/**
 * @brief initialize the tensor_demux's class
 */
static void
gst_tensor_demux_class_init (GstTensorDemuxClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->dispose = gst_tensor_demux_dispose;
  gobject_class->get_property = gst_tensor_demux_get_property;
  gobject_class->set_property = gst_tensor_demux_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_TENSORPICK,
      g_param_spec_int ("tensorpick", "TensorPick",
          "Choose nth tensor among tensors ?", 0, G_MAXINT, FALSE,
          G_PARAM_READWRITE));

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_demux_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "tensordemux",
      "Demux other/tensors stream",
      "Demux tensors stream to other/tensor stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_demux_init (GstTensorDemux * tensor_demux)
{
  tensor_demux->sinkpad =
      gst_pad_new_from_static_template (&sink_templ, "sink");
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_demux), tensor_demux->sinkpad);
  gst_pad_set_chain_function (tensor_demux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_demux_chain));
  gst_pad_set_event_function (tensor_demux->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_demux_event));

  tensor_demux->num_tensors = 0;
  tensor_demux->num_srcpads = 0;
  tensor_demux->silent = FALSE;
  tensor_demux->tensorpick = -1;
  tensor_demux->have_group_id = FALSE;
  tensor_demux->group_id = G_MAXUINT;
  tensor_demux->srcpads = NULL;
}

/**
 * @brief function to remove srcpad list for sink (gst element vmethod)
 */
static void
gst_tensor_demux_remove_src_pads (GstTensorDemux * tensor_demux)
{
  while (tensor_demux->srcpads != NULL) {
    GstTensorPad *tensor_pad = tensor_demux->srcpads->data;
    gst_element_remove_pad (GST_ELEMENT (tensor_demux), tensor_pad->pad);
    g_free (tensor_pad);
    tensor_demux->srcpads =
        g_slist_delete_link (tensor_demux->srcpads, tensor_demux->srcpads);
  }
  tensor_demux->srcpads = NULL;
  tensor_demux->num_tensors = 0;
  tensor_demux->num_srcpads = 0;
}

/**
 * @brief dispose function for tensor demux (gst element vmethod)
 */
static void
gst_tensor_demux_dispose (GObject * object)
{
  GstTensorDemux *tensor_demux = GST_TENSOR_DEMUX (object);

  gst_tensor_demux_remove_src_pads (tensor_demux);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief Set Caps in pad.
 * @param tensor_demux GstTensorDemux Ojbect
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully generate & set cap, return TRUE)
 */
static gboolean
gst_tensor_demux_get_capsparam (GstTensorDemux * tensor_demux, GstCaps * caps)
{
  gboolean ret = FALSE;

  GstStructure *s = gst_caps_get_structure (caps, 0);
  if (gst_structure_get_int (s, "num_tensors",
          (int *) &tensor_demux->num_tensors)
      && gst_structure_get_fraction (s, "framerate",
          &tensor_demux->framerate_numerator,
          &tensor_demux->framerate_denominator))
    ret = TRUE;

  return ret;
}

/**
 * @brief event function for sink (gst element vmethod)
 */
static gboolean
gst_tensor_demux_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorDemux *tensor_demux;
  tensor_demux = GST_TENSOR_DEMUX (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      gst_tensor_demux_get_capsparam (tensor_demux, caps);
      return gst_pad_event_default (pad, parent, event);
    }
    case GST_EVENT_EOS:
      if (!tensor_demux->srcpads) {
        GST_ELEMENT_ERROR (tensor_demux, STREAM, WRONG_TYPE,
            ("This stream contains no valid stremas."),
            ("Got EOS before adding any pads"));
        gst_event_unref (event);
        return FALSE;
      } else {
        return gst_pad_event_default (pad, parent, event);
      }
      break;
    default:
      return gst_pad_event_default (pad, parent, event);
  }
}

/**
 * @brief Checking if the source pad is created and if not, create TensorPad
 * @param tesnor_demux TensorDemux Object
 * @param inbuf inputbuf GstBuffer Object including GstMeta
 * @param[out] created will be updated in this function
 * @param nth source ordering
 * @return TensorPad if pad is already created, then return created pad.
 *         If not return new pad after creation.
 */
static GstTensorPad *
gst_get_tensor_pad (GstTensorDemux * tensor_demux, GstBuffer * inbuf,
    gboolean * created, gint nth)
{
  GSList *walk;
  walk = tensor_demux->srcpads;
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

  GstPad *pad;
  GstTensorPad *tensorpad;
  gchar *name;
  GstEvent *event;
  gchar *stream_id;
  GstCaps *caps;
  gchar *caps_string;
  tensor_dim *dim;
  tensor_type type;

  tensorpad = g_new0 (GstTensorPad, 1);
  GST_DEBUG_OBJECT (tensor_demux, "createing pad: %d(%dth)",
      tensor_demux->num_srcpads, nth);

  name = g_strdup_printf ("src_%u", tensor_demux->num_srcpads);
  pad = gst_pad_new_from_static_template (&src_templ, name);
  g_free (name);

  tensorpad->pad = pad;
  tensorpad->nth = nth;
  tensorpad->last_ret = GST_FLOW_OK;
  tensorpad->last_ts = GST_CLOCK_TIME_NONE;
  tensorpad->discont = TRUE;

  tensor_demux->srcpads = g_slist_append (tensor_demux->srcpads, tensorpad);
  dim = gst_get_tensordim (inbuf, tensor_demux->num_srcpads);
  type = gst_get_tensortype (inbuf, tensor_demux->num_srcpads);

  tensor_demux->num_srcpads++;

  gst_pad_use_fixed_caps (pad);
  gst_pad_set_active (pad, TRUE);


  if (!tensor_demux->have_group_id) {
    event =
        gst_pad_get_sticky_event (tensor_demux->sinkpad, GST_EVENT_STREAM_START,
        0);
    if (event) {
      tensor_demux->have_group_id =
          gst_event_parse_group_id (event, &tensor_demux->group_id);
      gst_event_unref (event);
    } else if (!tensor_demux->have_group_id) {
      tensor_demux->have_group_id = TRUE;
      tensor_demux->group_id = gst_util_group_id_next ();
    }
  }

  stream_id =
      gst_pad_create_stream_id (pad, GST_ELEMENT_CAST (tensor_demux),
      "other/tensors");

  event = gst_event_new_stream_start (stream_id);
  if (tensor_demux->have_group_id)
    gst_event_set_group_id (event, tensor_demux->group_id);

  gst_pad_store_sticky_event (pad, event);
  g_free (stream_id);
  gst_event_unref (event);

  caps_string = g_strdup_printf ("other/tensor, "
      "rank = (int)4, "
      "type = (string)%s,"
      "framerate = (fraction) %d/%d, "
      "dim1 = (int) %d, "
      "dim2 = (int) %d, "
      "dim3 = (int) %d, "
      "dim4 = (int) %d", tensor_element_typename[type],
      tensor_demux->framerate_numerator, tensor_demux->framerate_denominator,
      (*dim)[0], (*dim)[1], (*dim)[2], (*dim)[3]);

  caps = gst_caps_from_string (caps_string);
  GST_DEBUG_OBJECT (tensor_demux, "caps for pad : %s", caps_string);

  g_free (caps_string);
  gst_pad_set_caps (pad, caps);
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_demux), pad);

  gst_caps_unref (caps);

  if (created) {
    *created = TRUE;
  }

  if (tensor_demux->tensorpick != -1) {
    GST_DEBUG_OBJECT (tensor_demux, "TensorPick is set! : %dth tensor only\n",
        tensor_demux->tensorpick);
    gst_element_no_more_pads (GST_ELEMENT_CAST (tensor_demux));
  }

  return tensorpad;
}

/**
 * @brief Check the status among sources in demux
 * @param tensor_demux TensorDemux Object
 * @param TensorPad Tensorpad
 * @param ret return status of current pad
 * @return return status after check sources
 */
static GstFlowReturn
gst_tensordemux_combine_flows (GstTensorDemux * tensor_demux,
    GstTensorPad * pad, GstFlowReturn ret)
{
  GSList *walk;
  pad->last_ret = ret;

  if (ret != GST_FLOW_NOT_LINKED)
    goto done;

  for (walk = tensor_demux->srcpads; walk; walk = g_slist_next (walk)) {
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
gst_tensor_demux_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  gint num_tensors, i;
  GstFlowReturn res = GST_FLOW_OK;
  GstTensorDemux *tensor_demux;
  tensor_demux = GST_TENSOR_DEMUX (parent);

  if (GST_BUFFER_FLAG_IS_SET (buf, GST_BUFFER_FLAG_DISCONT)) {
    GSList *l;
    for (l = tensor_demux->srcpads; l != NULL; l = l->next) {
      GstTensorPad *srcpad = l->data;
      srcpad->discont = TRUE;
    }
  }

  num_tensors = gst_get_num_tensors (buf);

  GST_DEBUG_OBJECT (tensor_demux, " Number or Tensors: %d", num_tensors);

  for (i = 0; i < num_tensors; i++) {
    if (tensor_demux->tensorpick != -1 && tensor_demux->tensorpick != i) {
      continue;
    }

    GstTensorPad *srcpad;
    GstBuffer *outbuf;
    GstMemory *mem;
    gboolean created;
    GstClockTime ts;
    srcpad = gst_get_tensor_pad (tensor_demux, buf, &created, i);

    outbuf = gst_buffer_new ();
    mem = gst_get_tensor (buf, i);
    gst_buffer_append_memory (outbuf, mem);
    ts = GST_BUFFER_PTS (buf);

    if (created) {
      GstSegment segment;
      gst_segment_init (&segment, GST_FORMAT_TIME);
      gst_pad_push_event (srcpad->pad, gst_event_new_segment (&segment));
    }

    outbuf = gst_buffer_make_writable (outbuf);
    if (srcpad->last_ts == GST_CLOCK_TIME_NONE || srcpad->last_ts != ts) {
      GST_BUFFER_TIMESTAMP (outbuf) = ts;
      srcpad->last_ts = ts;
    } else {
      GST_BUFFER_TIMESTAMP (outbuf) = GST_CLOCK_TIME_NONE;
    }

    if (srcpad->discont) {
      GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DISCONT);
      srcpad->discont = FALSE;
    } else {
      GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_DISCONT);
    }

    GST_DEBUG_OBJECT (tensor_demux,
        "pushing buffer with timestamp %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)));
    res = gst_pad_push (srcpad->pad, outbuf);
    res = gst_tensordemux_combine_flows (tensor_demux, srcpad, res);

    if (res != GST_FLOW_OK)
      break;
  }

  return res;
}

/**
 * @brief change state (gst element vmethod)
 */
static GstStateChangeReturn
gst_tensor_demux_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorDemux *tensor_demux;
  GstStateChangeReturn ret;
  tensor_demux = GST_TENSOR_DEMUX (element);
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return ret;
  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      tensor_demux->group_id = G_MAXUINT;
      tensor_demux->have_group_id = FALSE;
      gst_tensor_demux_remove_src_pads (tensor_demux);
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
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
gst_tensor_demux_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorDemux *filter = GST_TENSOR_DEMUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_TENSORPICK:
      filter->tensorpick = g_value_get_int (value);
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
gst_tensor_demux_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorDemux *filter = GST_TENSOR_DEMUX (object);
  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_TENSORPICK:
      g_value_set_int (value, filter->tensorpick);
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
#define PACKAGE "tensordemux"
#endif

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
gboolean
gst_tensor_demux_plugin_init (GstPlugin * tensordemux)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_demux' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_demux_debug, "tensordemux", 0,
      "Tensor Demuxer");
  return gst_element_register (tensordemux, "tensordemux",
      GST_RANK_NONE, GST_TYPE_TENSOR_DEMUX);
}

/**
 * @brief gstreamer looks for this structure to register tensormux
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensordemux,
    "tensordemux",
    gst_tensor_demux_plugin_init, VERSION, "LGPL", "GStreamer",
    "http://gstreamer.net/");
