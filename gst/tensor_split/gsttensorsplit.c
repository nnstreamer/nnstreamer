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
 * @file	gsttensorsplit.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to split tensor (as a filter for other general neural network filters)
 * @bug         No known bugs
 *
 * @see		http://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensorsplit
 *
 * A Deuxer that split tensors stream to tensor stream for NN frameworks.
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
#include <stdlib.h>

#include "gsttensorsplit.h"
#include <tensor_meta.h>

GST_DEBUG_CATEGORY_STATIC (gst_tensor_split_debug);
#define GST_CAT_DEFAULT gst_tensor_split_debug

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_TENSORPICK,
  PROP_TENSORSEG
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
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static GstFlowReturn gst_tensor_split_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buf);
static gboolean gst_tensor_split_event (GstPad * pad, GstObject * parent,
    GstEvent * event);
static GstStateChangeReturn gst_tensor_split_change_state (GstElement * element,
    GstStateChange transition);
static void gst_tensor_split_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_split_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_split_dispose (GObject * object);
#define gst_tensor_split_parent_class parent_class
G_DEFINE_TYPE (GstTensorSplit, gst_tensor_split, GST_TYPE_ELEMENT);


/**
 * @brief initialize the tensor_split's class
 */
static void
gst_tensor_split_class_init (GstTensorSplitClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->dispose = gst_tensor_split_dispose;
  gobject_class->get_property = gst_tensor_split_get_property;
  gobject_class->set_property = gst_tensor_split_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_TENSORSEG,
      g_param_spec_boolean ("segment", "Segment", "How to split tensor ?",
          FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_TENSORPICK,
      g_param_spec_string ("tensorpick", "TensorPick",
          "Choose nth tensor among tensors ?", "", G_PARAM_READWRITE));

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_split_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "tensorsplit",
      "Split other/tensor stream",
      "Split tensor stream to other/tensor stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_split_init (GstTensorSplit * tensor_split)
{
  tensor_split->sinkpad =
      gst_pad_new_from_static_template (&sink_templ, "sink");
  gst_element_add_pad (GST_ELEMENT_CAST (tensor_split), tensor_split->sinkpad);
  gst_pad_set_chain_function (tensor_split->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_split_chain));
  gst_pad_set_event_function (tensor_split->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_split_event));

  tensor_split->num_tensors = 0;
  tensor_split->num_srcpads = 0;
  tensor_split->silent = FALSE;
  tensor_split->tensorpick = NULL;
  tensor_split->tensorseg = NULL;
  tensor_split->have_group_id = FALSE;
  tensor_split->group_id = G_MAXUINT;
  tensor_split->srcpads = NULL;
}

/**
 * @brief function to remove srcpad list
 */
static void
gst_tensor_split_remove_src_pads (GstTensorSplit * tensor_split)
{
  while (tensor_split->srcpads != NULL) {
    GstTensorPad *tensor_pad = tensor_split->srcpads->data;
    gst_element_remove_pad (GST_ELEMENT (tensor_split), tensor_pad->pad);
    g_free (tensor_pad);
    tensor_split->srcpads =
        g_slist_delete_link (tensor_split->srcpads, tensor_split->srcpads);
  }
  tensor_split->srcpads = NULL;
  tensor_split->num_tensors = 0;
  tensor_split->num_srcpads = 0;
}

/**
 * @brief dispose function for tensor split (gst element vmethod)
 */
static void
gst_tensor_split_dispose (GObject * object)
{
  GstTensorSplit *tensor_split = GST_TENSOR_SPLIT (object);

  gst_tensor_split_remove_src_pads (tensor_split);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief Set Caps in pad.
 * @param tensor_split GstTensorSplit Ojbect
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully generate & set cap, return TRUE)
 */
static gboolean
gst_tensor_split_get_capsparam (GstTensorSplit * tensor_split, GstCaps * caps)
{
  gboolean ret = FALSE;

  return ret;
}

/**
 * @brief event function for sink (gst element vmethod)
 */
static gboolean
gst_tensor_split_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorSplit *tensor_split;
  tensor_split = GST_TENSOR_SPLIT (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      gst_tensor_split_get_capsparam (tensor_split, caps);
      return gst_pad_event_default (pad, parent, event);
    }
    case GST_EVENT_EOS:
      if (!tensor_split->srcpads) {
        GST_ELEMENT_ERROR (tensor_split, STREAM, WRONG_TYPE,
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
 * @brief chain function for sink (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_split_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstFlowReturn res = GST_FLOW_OK;

  return res;
}

/**
 * @brief change state (gst element vmethod)
 */
static GstStateChangeReturn
gst_tensor_split_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorSplit *tensor_split;
  GstStateChangeReturn ret;
  tensor_split = GST_TENSOR_SPLIT (element);
  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return ret;
  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      tensor_split->group_id = G_MAXUINT;
      tensor_split->have_group_id = FALSE;
      gst_tensor_split_remove_src_pads (tensor_split);
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
gst_tensor_split_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorSplit *self = GST_TENSOR_SPLIT (object);
  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
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
        self->tensorpick =
            g_list_append (self->tensorpick, GINT_TO_POINTER (val));
      }
      break;
    }
    case PROP_TENSORSEG:
    {
      gint i;
      const gchar *param = g_value_get_string (value);
      gchar **strv = g_strsplit_set (param, ",.;/", -1);
      self->num_tensors = g_strv_length (strv);
      self->tensorseg =
          g_array_sized_new (FALSE, FALSE, sizeof (tensor_dim *),
          self->num_tensors);
      for (i = 0; i < self->num_tensors; i++) {
        gchar **p;
        gint num, k;
        tensor_dim *d;
        p = g_strsplit_set (strv[i], ":", -1);
        num = g_strv_length (p);
        d = g_new0 (tensor_dim, 1);
        for (k = 0; k < num; k++) {
          (*d)[k] = atoi (p[k]);
        }
        g_array_append_val (self->tensorseg, d);
        g_strfreev (p);
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
gst_tensor_split_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorSplit *filter = GST_TENSOR_SPLIT (object);
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
    case PROP_TENSORSEG:
    {
      break;
    }

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
#define PACKAGE "tensorsplit"
#endif

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
gboolean
gst_tensor_split_plugin_init (GstPlugin * tensorsplit)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_split' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_split_debug, "tensorsplit", 0,
      "Tensor Spliter");
  return gst_element_register (tensorsplit, "tensorsplit",
      GST_RANK_NONE, GST_TYPE_TENSOR_SPLIT);
}

/**
 * @brief gstreamer looks for this structure to register tensorsplit
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensorsplit,
    "tensorsplit",
    gst_tensor_split_plugin_init, VERSION, "LGPL", "GStreamer",
    "http://gstreamer.net/");
