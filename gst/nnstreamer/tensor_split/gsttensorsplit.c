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
 * @date	27 Aug 2018
 * @brief	GStreamer plugin to split tensor (as a filter for other general neural network filters)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensor_split
 *
 * A Deuxer that split tensors stream to tensor stream for NN frameworks.
 * The outputs are always in the format of other/tensor.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1 ! tensor_converter
 * ! tensor_split name=split tensorseg=2:100:100,1:100:100 split.src_0 ! queue ! filesink location=src0.log
 * split.src_1 ! queue ! filesink location=src1.log
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

#include "gsttensorsplit.h"
#include <tensor_common.h>

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

  GST_DEBUG_CATEGORY_INIT (gst_tensor_split_debug, "tensor_split", 0,
      "Element to split tensors stream to tensor stream");

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->dispose = gst_tensor_split_dispose;
  gobject_class->get_property = gst_tensor_split_get_property;
  gobject_class->set_property = gst_tensor_split_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Do not produce verbose output ?", TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_TENSORPICK,
      g_param_spec_string ("tensorpick", "TensorPick",
          "Choose nth tensor among tensors ?", "", G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_TENSORSEG,
      g_param_spec_string ("tensorseg", "TensorSeg",
          "How to split tensor ?", "", G_PARAM_READWRITE));

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_split_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorSplit",
      "Demuxer/Tensor",
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
gst_tensor_split_init (GstTensorSplit * split)
{
  split->sinkpad = gst_pad_new_from_static_template (&sink_templ, "sink");
  gst_element_add_pad (GST_ELEMENT_CAST (split), split->sinkpad);
  gst_pad_set_chain_function (split->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_split_chain));
  gst_pad_set_event_function (split->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_split_event));

  split->num_tensors = 0;
  split->num_srcpads = 0;
  split->silent = TRUE;
  split->tensorpick = NULL;
  split->tensorseg = NULL;
  split->have_group_id = FALSE;
  split->group_id = G_MAXUINT;
  split->srcpads = NULL;
  gst_tensor_config_init (&split->sink_tensor_conf);
}

/**
 * @brief function to remove srcpad list
 */
static void
gst_tensor_split_remove_src_pads (GstTensorSplit * split)
{
  while (split->srcpads != NULL) {
    GstTensorPad *tensor_pad = split->srcpads->data;
    gst_element_remove_pad (GST_ELEMENT (split), tensor_pad->pad);
    g_free (tensor_pad);
    split->srcpads = g_slist_delete_link (split->srcpads, split->srcpads);
  }
  split->srcpads = NULL;
  split->num_tensors = 0;
  split->num_srcpads = 0;
}

/**
 * @brief dispose function for tensor split (gst element vmethod)
 */
static void
gst_tensor_split_dispose (GObject * object)
{
  GstTensorSplit *split;

  split = GST_TENSOR_SPLIT (object);
  gst_tensor_split_remove_src_pads (split);

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

/**
 * @brief Set Caps in pad.
 * @param split GstTensorSplit Ojbect
 * @param caps incomming capablity
 * @return TRUE/FALSE (if successfully generate & set cap, return TRUE)
 */
static gboolean
gst_tensor_split_get_capsparam (GstTensorSplit * split, GstCaps * caps)
{
  GstStructure *st;

  st = gst_caps_get_structure (caps, 0);

  return gst_tensor_config_from_structure (&split->sink_tensor_conf, st);
}

/**
 * @brief event function for sink (gst element vmethod)
 */
static gboolean
gst_tensor_split_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstTensorSplit *split;

  split = GST_TENSOR_SPLIT (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gst_event_parse_caps (event, &caps);
      if (!gst_tensor_split_get_capsparam (split, caps)) {
        GST_ELEMENT_ERROR (split, STREAM, WRONG_TYPE,
            ("This stream contains no valid type."), NULL);
      }
      break;
    }
    case GST_EVENT_EOS:
      if (!split->srcpads) {
        GST_ELEMENT_ERROR (split, STREAM, WRONG_TYPE,
            ("This stream contains no valid stremas."),
            ("Got EOS before adding any pads"));
        gst_event_unref (event);
        return FALSE;
      }
      break;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief Checking if the source pad is created and if not, create TensorPad
 * @param split TensorSplit Object
 * @param inbuf inputbuf GstBuffer Object including GstMeta
 * @param[out] created will be updated in this function
 * @param nth source ordering
 * @return TensorPad if pad is already created, then return created pad.
 *         If not return new pad after creation.
 */
static GstTensorPad *
gst_tensor_split_get_tensor_pad (GstTensorSplit * split, GstBuffer * inbuf,
    gboolean * created, gint nth)
{
  GSList *walk;
  GstPad *pad;
  GstTensorPad *tensorpad;
  gchar *name;
  GstEvent *event;
  gchar *stream_id;
  GstCaps *caps;
  GstTensorConfig pad_config;
  tensor_dim *dim;
  guint i;

  walk = split->srcpads;
  while (walk) {
    GstTensorPad *pad = (GstTensorPad *) walk->data;
    if (nth == pad->nth) {
      if (created) {
        *created = FALSE;
      }
      return pad;
    }
    walk = g_slist_next (walk);
  }

  tensorpad = g_new0 (GstTensorPad, 1);
  g_assert (tensorpad != NULL);
  GST_DEBUG_OBJECT (split, "createing pad: %d(%dth)", split->num_srcpads, nth);

  name = g_strdup_printf ("src_%u", split->num_srcpads);
  pad = gst_pad_new_from_static_template (&src_templ, name);
  g_free (name);

  tensorpad->pad = pad;
  tensorpad->nth = nth;
  tensorpad->last_ret = GST_FLOW_OK;
  tensorpad->last_ts = GST_CLOCK_TIME_NONE;

  split->srcpads = g_slist_append (split->srcpads, tensorpad);
  dim = g_array_index (split->tensorseg, tensor_dim *, split->num_srcpads);

  split->num_srcpads++;

  gst_pad_use_fixed_caps (pad);
  gst_pad_set_active (pad, TRUE);

  if (!split->have_group_id) {
    event =
        gst_pad_get_sticky_event (split->sinkpad, GST_EVENT_STREAM_START, 0);
    if (event) {
      split->have_group_id = gst_event_parse_group_id (event, &split->group_id);
      gst_event_unref (event);
    } else if (!split->have_group_id) {
      split->have_group_id = TRUE;
      split->group_id = gst_util_group_id_next ();
    }
  }

  stream_id =
      gst_pad_create_stream_id (pad, GST_ELEMENT_CAST (split), "other/tensor");

  event = gst_event_new_stream_start (stream_id);
  if (split->have_group_id)
    gst_event_set_group_id (event, split->group_id);

  gst_pad_store_sticky_event (pad, event);
  g_free (stream_id);
  gst_event_unref (event);

  /* tensor config to set caps */
  gst_tensor_config_init (&pad_config);

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    pad_config.info.dimension[i] = (*dim)[i];
  }
  pad_config.info.type = split->sink_tensor_conf.info.type;
  pad_config.rate_n = split->sink_tensor_conf.rate_n;
  pad_config.rate_d = split->sink_tensor_conf.rate_d;

  caps = gst_tensor_caps_from_config (&pad_config);

  gst_pad_set_caps (pad, caps);
  gst_element_add_pad (GST_ELEMENT_CAST (split), pad);

  gst_caps_unref (caps);

  if (created) {
    *created = TRUE;
  }

  if (split->tensorpick != NULL) {
    GST_DEBUG_OBJECT (split, "TensorPick is set! : %dth tensor\n", nth);
    if (g_list_length (split->tensorpick) == split->num_srcpads) {
      gst_element_no_more_pads (GST_ELEMENT_CAST (split));
    }
  }

  return tensorpad;
}

/**
 * @brief Check the status among sources in split
 * @param split TensorSplit Object
 * @param TensorPad Tensorpad
 * @param ret return status of current pad
 * @return return status after check sources
 */
static GstFlowReturn
gst_tensor_split_combine_flows (GstTensorSplit * split,
    GstTensorPad * pad, GstFlowReturn ret)
{
  GSList *walk;
  GstTensorPad *opad;

  pad->last_ret = ret;
  if (ret != GST_FLOW_NOT_LINKED)
    goto done;

  for (walk = split->srcpads; walk; walk = g_slist_next (walk)) {
    opad = (GstTensorPad *) walk->data;
    ret = opad->last_ret;
    if (ret != GST_FLOW_NOT_LINKED)
      goto done;
  }
done:
  return ret;
}

/**
 * @brief Make Splited Tensor
 * @param split TensorSplit Object
 * @param buffer gstbuffer form src
 * @param nth orther of tensor
 * @return return GstMemory for splited tensor
 */
static GstMemory *
gst_tensor_split_get_splited (GstTensorSplit * split, GstBuffer * buffer,
    gint nth)
{
  GstMemory *mem;
  tensor_dim *dim;
  int i;
  size_t size, offset;
  GstMapInfo src_info, dest_info;

  size = 0;
  offset = 0;
  dim = g_array_index (split->tensorseg, tensor_dim *, nth);

  size += gst_tensor_get_element_count (*dim) *
      gst_tensor_get_element_size (split->sink_tensor_conf.info.type);
  mem = gst_allocator_alloc (NULL, size, NULL);
  g_assert (gst_memory_map (mem, &dest_info, GST_MAP_WRITE));
  g_assert (gst_buffer_map (buffer, &src_info, GST_MAP_READ));

  for (i = 0; i < nth; i++) {
    dim = g_array_index (split->tensorseg, tensor_dim *, i);
    offset += gst_tensor_get_element_count (*dim);
  }

  nns_memcpy (dest_info.data, src_info.data + offset, size);
  gst_buffer_unmap (buffer, &src_info);
  gst_memory_unmap (mem, &dest_info);

  return mem;
}

/**
 * @brief chain function for sink (gst element vmethod)
 */
static GstFlowReturn
gst_tensor_split_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstTensorSplit *split;
  gint num_tensors, i;
  GstFlowReturn res = GST_FLOW_OK;

  split = GST_TENSOR_SPLIT (parent);

  num_tensors = split->num_tensors;
  GST_DEBUG_OBJECT (split, " Number of Tensors: %d", num_tensors);

  if (split->tensorseg == NULL) {
    GST_ERROR_OBJECT (split, "No rule to split incoming buffers.");
    return GST_FLOW_ERROR;
  }

  for (i = 0; i < num_tensors; i++) {
    GstTensorPad *srcpad;
    GstBuffer *outbuf;
    GstMemory *mem;
    gboolean created;
    GstClockTime ts;

    if (split->tensorpick != NULL) {
      gboolean found = FALSE;
      GList *list;
      for (list = split->tensorpick; list != NULL; list = list->next) {
        if (i == GPOINTER_TO_INT (list->data)) {
          found = TRUE;
          break;
        }
      }
      if (!found)
        continue;
    }

    srcpad = gst_tensor_split_get_tensor_pad (split, buf, &created, i);

    outbuf = gst_buffer_new ();
    mem = gst_tensor_split_get_splited (split, buf, i);
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
      GST_DEBUG_OBJECT (split, "invalid timestamp %" GST_TIME_FORMAT,
          GST_TIME_ARGS (ts));
    }

    GST_DEBUG_OBJECT (split, "pushing buffer with timestamp %" GST_TIME_FORMAT,
        GST_TIME_ARGS (GST_BUFFER_TIMESTAMP (outbuf)));
    res = gst_pad_push (srcpad->pad, outbuf);
    res = gst_tensor_split_combine_flows (split, srcpad, res);
    if (res != GST_FLOW_OK)
      break;
  }

  return res;
}

/**
 * @brief change state (gst element vmethod)
 */
static GstStateChangeReturn
gst_tensor_split_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorSplit *split;
  GstStateChangeReturn ret;

  split = GST_TENSOR_SPLIT (element);

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE)
    return ret;

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      split->group_id = G_MAXUINT;
      split->have_group_id = FALSE;
      gst_tensor_split_remove_src_pads (split);
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
  GstTensorSplit *split;

  split = GST_TENSOR_SPLIT (object);

  switch (prop_id) {
    case PROP_SILENT:
      split->silent = g_value_get_boolean (value);
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
        split->tensorpick =
            g_list_append (split->tensorpick, GINT_TO_POINTER (val));
      }
      g_strfreev (strv);
      break;
    }
    case PROP_TENSORSEG:
    {
      gint i;
      const gchar *param = g_value_get_string (value);
      gchar **strv = g_strsplit_set (param, ",.;/", -1);
      split->num_tensors = g_strv_length (strv);
      split->tensorseg =
          g_array_sized_new (FALSE, FALSE, sizeof (tensor_dim *),
          split->num_tensors);
      for (i = 0; i < split->num_tensors; i++) {
        gchar **p;
        gint num, k;
        tensor_dim *d;
        p = g_strsplit_set (strv[i], ":", -1);
        num = g_strv_length (p);
        d = g_new0 (tensor_dim, 1);
        g_assert (d != NULL);
        for (k = 0; k < num; k++) {
          (*d)[k] = g_ascii_strtod (p[k], NULL);
        }
        for (k = num; k < NNS_TENSOR_RANK_LIMIT; k++)
          (*d)[k] = 1;

        g_array_append_val (split->tensorseg, d);
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
  GstTensorSplit *split;

  split = GST_TENSOR_SPLIT (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, split->silent);
      break;
    case PROP_TENSORPICK:
    {
      GList *list;
      char *p;
      GPtrArray *arr = g_ptr_array_new ();
      gchar **strings;

      for (list = split->tensorpick; list != NULL; list = list->next) {
        g_ptr_array_add (arr, g_strdup_printf ("%i",
                GPOINTER_TO_INT (list->data)));
      }
      g_ptr_array_add (arr, NULL);
      strings = (gchar **) g_ptr_array_free (arr, FALSE);
      p = g_strjoinv (",", strings);
      g_strfreev (strings);
      g_value_take_string (value, p);
      break;
    }
    case PROP_TENSORSEG:
    {
      if (split->tensorseg && split->tensorseg->len > 0) {
        tensor_dim *dim = NULL;
        int i, j;
        gchar **strings;
        gchar *p, *strv;

        for (i = 0; i < split->tensorseg->len; i++) {
          GPtrArray *arr = g_ptr_array_new ();
          dim = g_array_index (split->tensorseg, tensor_dim *, i);
          for (j = 0; j < NNS_TENSOR_RANK_LIMIT; j++) {
            g_ptr_array_add (arr, g_strdup_printf ("%i", (*dim)[j]));
          }
          g_ptr_array_add (arr, NULL);
          strings = (gchar **) g_ptr_array_free (arr, FALSE);
          p = g_strjoinv (":", strings);
          g_strfreev (strings);
          if (i > 0) {
            /**
             * If i = 1, this is previous p.
             * Otherwise, it's previous g_strjoin result.
             */
            gchar *oldstrv = strv;

            strv = g_strjoin (",", strv, p, NULL);
            g_free (oldstrv);
            g_free (p);
          } else {
            strv = p;
          }
        }
        g_value_take_string (value, strv);
      } else {
        g_value_set_string (value, "");
      }
      break;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}
