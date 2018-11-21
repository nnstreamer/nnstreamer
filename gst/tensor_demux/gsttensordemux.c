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
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensor_demux
 *
 * A Deuxer that demux tensors stream to tensor stream for NN frameworks.
 * The outputs are always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 tensor_mux name=mux ! tensor_demux name=demux \
 * filesrc location=testcase01_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 \
 * filesrc location=testcase01_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 \
 * filesrc location=testcase01_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 \
 * demux.src_0 ! queue ! filesink location=demux00.log \
 * demux.src_1 ! queue ! filesink location=demux01.log \
 * demux.src_2 ! queue ! filesink location=demux02.log
 * ]|
 *
 * |[
 * gst-launch-1.0 tensor_mux name=mux ! tensor_demux name=demux \
 * multifilesrc location="testsequence01_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_0 \
 * multifilesrc location="testsequence01_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_1 \
 * multifilesrc location="testsequence01_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! mux.sink_2 \
 * demux.src_0 ! queue ! filesink location=demux00.log \
 * demux.src_1 ! queue ! filesink location=demux01.log \
 * demux.src_2 ! queue ! filesink location=demux02.log
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
          TRUE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TENSORPICK,
      g_param_spec_string ("tensorpick", "TensorPick",
          "Choose nth tensor among tensors ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_demux_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorDemux",
      "Demuxer/Tensor",
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

  tensor_demux->num_srcpads = 0;
  tensor_demux->silent = TRUE;
  tensor_demux->tensorpick = NULL;
  tensor_demux->have_group_id = FALSE;
  tensor_demux->group_id = G_MAXUINT;
  tensor_demux->srcpads = NULL;

  gst_tensors_config_init (&tensor_demux->tensors_config);
}

/**
 * @brief function to remove srcpad list
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
  tensor_demux->num_srcpads = 0;

  gst_tensors_config_init (&tensor_demux->tensors_config);
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
 * @brief Parse caps and configure tensors info.
 * @param tensor_demux GstTensorDemux Ojbect
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully configured, return TRUE)
 */
static gboolean
gst_tensor_demux_parse_caps (GstTensorDemux * tensor_demux, GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsConfig *config;

  config = &tensor_demux->tensors_config;

  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (config, structure);

  return gst_tensors_config_validate (config);
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
      gst_tensor_demux_parse_caps (tensor_demux, caps);
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
 * @brief Get tensor config info from configured tensors
 * @param tensor_demux "this" pointer
 * @param config tensor config to be filled
 * @param index index of configured tensors
 * @return
 */
static gboolean
gst_tensor_demux_get_tensor_config (GstTensorDemux * tensor_demux,
    GstTensorConfig * config, guint index)
{
  GstTensorsConfig *tensors_info;

  g_return_val_if_fail (tensor_demux != NULL, FALSE);
  g_return_val_if_fail (config != NULL, FALSE);

  gst_tensor_config_init (config);

  tensors_info = &tensor_demux->tensors_config;
  g_return_val_if_fail (index < tensors_info->info.num_tensors, FALSE);

  config->info = tensors_info->info.info[index];
  config->rate_n = tensors_info->rate_n;
  config->rate_d = tensors_info->rate_d;
  return TRUE;
}

/**
 * @brief Checking if the source pad is created and if not, create TensorPad
 * @param tesnor_demux TensorDemux Object
 * @param[out] created will be updated in this function
 * @param nth source ordering
 * @return TensorPad if pad is already created, then return created pad.
 *         If not return new pad after creation.
 */
static GstTensorPad *
gst_tensor_demux_get_tensor_pad (GstTensorDemux * tensor_demux,
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
  GstTensorConfig config;

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

  tensor_demux->srcpads = g_slist_append (tensor_demux->srcpads, tensorpad);
  gst_tensor_demux_get_tensor_config (tensor_demux, &config,
      tensor_demux->num_srcpads);

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

  caps = gst_tensor_caps_from_config (&config);
  gst_pad_set_caps (pad, caps);
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_demux), pad);

  gst_caps_unref (caps);

  if (created) {
    *created = TRUE;
  }

  if (tensor_demux->tensorpick != NULL) {
    GST_DEBUG_OBJECT (tensor_demux, "TensorPick is set! : %dth tensor\n", nth);
    if (g_list_length (tensor_demux->tensorpick) == tensor_demux->num_srcpads) {
      gst_element_no_more_pads (GST_ELEMENT_CAST (tensor_demux));
    }
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
gst_tensor_demux_combine_flows (GstTensorDemux * tensor_demux,
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

  num_tensors = tensor_demux->tensors_config.info.num_tensors;
  GST_DEBUG_OBJECT (tensor_demux, " Number of Tensors: %d", num_tensors);

  /* supposed n memory blocks in buffer */
  g_assert (gst_buffer_n_memory (buf) == num_tensors);

  for (i = 0; i < num_tensors; i++) {
    if (tensor_demux->tensorpick != NULL) {
      gboolean found = FALSE;
      GList *list;
      for (list = tensor_demux->tensorpick; list != NULL; list = list->next) {
        if (i == GPOINTER_TO_INT (list->data)) {
          found = TRUE;
          break;
        }
      }
      if (!found)
        continue;
    }

    GstTensorPad *srcpad;
    GstBuffer *outbuf;
    GstMemory *mem;
    gboolean created;
    GstClockTime ts;
    srcpad = gst_tensor_demux_get_tensor_pad (tensor_demux, &created, i);

    outbuf = gst_buffer_new ();
    mem = gst_buffer_peek_memory (buf, i);
    gst_buffer_append_memory (outbuf, mem);
    ts = GST_BUFFER_TIMESTAMP (buf);

    if (created) {
      GstSegment segment;
      gst_segment_init (&segment, GST_FORMAT_TIME);
      gst_pad_push_event (srcpad->pad, gst_event_new_segment (&segment));
    }

    outbuf = gst_buffer_make_writable (outbuf);

    /* metadata from incoming buffer */
    gst_buffer_copy_into (outbuf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

    if (srcpad->last_ts == GST_CLOCK_TIME_NONE || srcpad->last_ts != ts) {
      srcpad->last_ts = ts;
    } else {
      GST_DEBUG_OBJECT (tensor_demux, "invalid timestamp %" GST_TIME_FORMAT,
          GST_TIME_ARGS (ts));
    }

    GST_DEBUG_OBJECT (tensor_demux,
        "pushing buffer with timestamp %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)));
    res = gst_pad_push (srcpad->pad, outbuf);
    res = gst_tensor_demux_combine_flows (tensor_demux, srcpad, res);

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
    {
      gint i;
      gint64 val;
      const gchar *param = g_value_get_string (value);
      gchar **strv = g_strsplit_set (param, ",.;/", -1);
      gint num = g_strv_length (strv);
      for (i = 0; i < num; i++) {
        val = g_ascii_strtoll (strv[i], NULL, 10);
        filter->tensorpick =
            g_list_append (filter->tensorpick, GINT_TO_POINTER (val));
      }
      g_strfreev (strv);
      break;
    }
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
    {
      GList *list;
      char *p = "";
      GPtrArray *arr = g_ptr_array_new ();
      gchar **strings;

      for (list = filter->tensorpick; list != NULL; list = list->next) {
        g_ptr_array_add (arr, g_strdup_printf ("%i",
                GPOINTER_TO_INT (list->data)));
      }
      g_ptr_array_add (arr, NULL);
      strings = (gchar **) g_ptr_array_free (arr, FALSE);
      p = g_strjoinv (",", strings);
      g_free (strings);
      g_value_set_string (value, p);
      break;
    }
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
NNSTREAMER_PLUGIN_INIT (tensor_demux)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_demux' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_demux_debug, "tensor_demux", 0,
      "Tensor Demuxer");
  return gst_element_register (plugin, "tensor_demux",
      GST_RANK_NONE, GST_TYPE_TENSOR_DEMUX);
}
