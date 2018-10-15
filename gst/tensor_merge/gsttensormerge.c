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
 * @file	gsttensormerge.c
 * @date	03 July 2018
 * @brief	GStreamer plugin to merge tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

/**
 * SECTION:element-tensormerge
 *
 * A Merger that merge tensor stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m tensor_merge name=merge ! fakesink
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_0
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_1
 * filesrc location=b.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! merge.sink_2
 * ]|
 *
 * |[
 * gst-launch -v -m tensor_merge name=merge ! filesink location=merge.log
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! merge.sink_0
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! merge.sink_1
 * multifilesrc location="testsequence_%1d.png" index=0 caps="image/png, framerate=(fraction)30/1" ! pngdec ! tensor_converter ! merge.sink_2
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

#include "gsttensormerge.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_merge_debug);
#define GST_CAT_DEFAULT gst_tensor_merge_debug

enum
{
  PROP_0,
  PROP_MODE,
  PROP_OPTION,
  PROP_SILENT,
};

/**
 * @brief the capabilities of the inputs and outputs.
 * describe the real formats here.
 */
static GstStaticPadTemplate src_templ = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static GstStaticPadTemplate sink_templ = GST_STATIC_PAD_TEMPLATE ("sink_%u",
    GST_PAD_SINK,
    GST_PAD_REQUEST,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT)
    );

static gboolean gst_tensor_merge_handle_src_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static GstPad *gst_tensor_merge_request_new_pad (GstElement * element,
    GstPadTemplate * templ, const gchar * name, const GstCaps * caps);
static GstStateChangeReturn gst_tensor_merge_change_state (GstElement * element,
    GstStateChange transition);
static gboolean gst_tensor_merge_sink_event (GstCollectPads * pads,
    GstCollectData * data, GstEvent * event, GstTensorMerge * tensor_merge);
static GstFlowReturn gst_tensor_merge_collected (GstCollectPads * pads,
    GstTensorMerge * tensor_merge);

static void gst_tensor_merge_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_merge_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_merge_finalize (GObject * object);

#define gst_tensor_merge_parent_class parent_class
G_DEFINE_TYPE (GstTensorMerge, gst_tensor_merge, GST_TYPE_ELEMENT);

/**
 * @brief initialize the tensor_merge's class
 */
static void
gst_tensor_merge_class_init (GstTensorMergeClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  parent_class = g_type_class_peek_parent (klass);

  gobject_class->finalize = gst_tensor_merge_finalize;
  gobject_class->get_property = gst_tensor_merge_get_property;
  gobject_class->set_property = gst_tensor_merge_set_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          TRUE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_string ("mode", "Mode", "Tensor transform mode ?",
          "", G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_OPTION,
      g_param_spec_string ("option", "Option",
          "Option for the tensor transform mode ?", "", G_PARAM_READWRITE));

  gstelement_class->request_new_pad =
      GST_DEBUG_FUNCPTR (gst_tensor_merge_request_new_pad);
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_tensor_merge_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_templ));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_templ));

  gst_element_class_set_details_simple (gstelement_class,
      "TensorMerge",
      "Merger/Tensor",
      "Merge multiple tensor stream to tensor stream",
      "Jijoong Moon <jijoong.moon@samsung.com>");

}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_merge_init (GstTensorMerge * tensor_merge)
{
  GstElementClass *klass = GST_ELEMENT_GET_CLASS (tensor_merge);

  tensor_merge->srcpad =
      gst_pad_new_from_template (gst_element_class_get_pad_template (klass,
          "src"), "src");
  gst_pad_set_event_function (tensor_merge->srcpad,
      gst_tensor_merge_handle_src_event);

  gst_element_add_pad (GST_ELEMENT (tensor_merge), tensor_merge->srcpad);

  tensor_merge->collect = gst_collect_pads_new ();
  gst_collect_pads_set_event_function (tensor_merge->collect,
      (GstCollectPadsEventFunction)
      GST_DEBUG_FUNCPTR (gst_tensor_merge_sink_event), tensor_merge);
  gst_collect_pads_set_function (tensor_merge->collect,
      (GstCollectPadsFunction) GST_DEBUG_FUNCPTR (gst_tensor_merge_collected),
      tensor_merge);

  tensor_merge->silent = TRUE;
  gst_tensors_config_init (&tensor_merge->tensors_config);
  tensor_merge->mode = GTT_END;
  tensor_merge->option = NULL;
  tensor_merge->loaded = FALSE;
}

static const gchar *gst_tensor_merge_mode_string[] = {
  [GTT_LINEAR] = "linear",
  [GTT_END] = "error",
};

static const gchar *gst_tensor_merge_linear_string[] = {
  [LINEAR_FIRST] = "0",
  [LINEAR_SECOND] = "1",
  [LINEAR_THIRD] = "2",
  [LINEAR_FOURTH] = "3",
  [LINEAR_END] = NULL,
};

/**
 * @brief Get the corresponding mode from the string value
 * @param[in] str The string value for the mode
 * @return corresponding mode for the string. GTT_END for errors
 */
static tensor_merge_mode
gst_tensor_merge_get_mode (const gchar * str)
{
  int i;
  for (i = 0; i < GTT_END; i++) {
    if (!g_ascii_strcasecmp (gst_tensor_merge_mode_string[i], str))
      return i;
  }
  return GTT_END;
}

/**
 * @brief finalize vmethod
 */
static void
gst_tensor_merge_finalize (GObject * object)
{
  GstTensorMerge *tensor_merge;
  tensor_merge = GST_TENSOR_MERGE (object);

  if (tensor_merge->collect) {
    gst_object_unref (tensor_merge->collect);
    tensor_merge->collect = NULL;
  }

  if (tensor_merge->option) {
    g_free (tensor_merge->option);
    tensor_merge->option = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief making new request pad (gst element vmethod)
 */
static GstPad *
gst_tensor_merge_request_new_pad (GstElement * element, GstPadTemplate * templ,
    const gchar * req_name, const GstCaps * caps)
{
  GstPad *newpad;
  GstTensorMerge *tensor_merge;
  gchar *name;

  g_return_val_if_fail (templ != NULL, NULL);
  g_return_val_if_fail (GST_IS_TENSOR_MERGE (element), NULL);

  tensor_merge = GST_TENSOR_MERGE (element);

  name =
      g_strdup_printf ("sink_%u",
      tensor_merge->tensors_config.info.num_tensors);
  newpad = gst_pad_new_from_template (templ, name);
  g_free (name);

  if (newpad) {

    GstTensorMergePadData *tensormergepad;

    tensormergepad = (GstTensorMergePadData *)
        gst_collect_pads_add_pad (tensor_merge->collect, newpad,
        sizeof (GstTensorMergePadData), NULL, TRUE);

    tensormergepad->pad = newpad;
    gst_pad_set_element_private (newpad, tensormergepad);
    tensor_merge->tensors_config.info.num_tensors++;
    gst_element_add_pad (element, newpad);
  } else {
    GST_WARNING_OBJECT (tensor_merge, "failed to create request pad");
  }
  return newpad;
}

/**
 * @brief src event vmethod
 */
static gboolean
gst_tensor_merge_handle_src_event (GstPad * pad, GstObject * parent,
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
gst_tensor_merge_sink_event (GstCollectPads * pads, GstCollectData * data,
    GstEvent * event, GstTensorMerge * tensor_merge)
{
  gboolean ret;
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_STOP:
    {
      tensor_merge->need_segment = TRUE;
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
 * @param tensor_merge tensor merger
 * @param old_data previous merge pad data
 * @param new_data current merge pad data
 * @return if > 0, new is earlier than old
 */
static gint
gst_tensor_merge_compare_pads (GstTensorMerge * tensor_merge,
    GstTensorMergePadData * old_data, GstTensorMergePadData * new_data)
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
 * @brief Generate TensorConfig with TensorsConfig
 * @param tensor_merge tensor merger
 * @param configs Tensors Config Data
 * @param config Tensor Config Data
 * @return true / false
 */
gboolean
gst_merge_tensors_config (GstTensorMerge * tensor_merge,
    GstTensorsConfig * configs, GstTensorConfig * config)
{
  gboolean ret = FALSE;
  int i, j;
  tensor_dim dim;
  tensor_type type;
  type = configs->info.info[0].type;
  memcpy (&dim, &configs->info.info[0].dimension, sizeof (tensor_dim));

  for (i = 1; i < configs->info.num_tensors; i++) {
    if (type != configs->info.info[i].type)
      GST_ELEMENT_ERROR (tensor_merge, CORE, NEGOTIATION, (NULL), (NULL));
  }

  switch (tensor_merge->mode) {
    case GTT_LINEAR:
    {
      int targetIdx = tensor_merge->data_linear.direction;
      for (i = 1; i < configs->info.num_tensors; i++) {
        for (j = 0; j < NNS_TENSOR_RANK_LIMIT; j++) {
          if (j == targetIdx) {
            dim[j] += configs->info.info[i].dimension[j];
          } else {
            if (dim[j] != configs->info.info[i].dimension[j])
              GST_ELEMENT_ERROR (tensor_merge, CORE, NEGOTIATION, (NULL),
                  (NULL));
          }
        }
      }
      config->info.type = type;
      memcpy (&config->info.dimension, &dim, sizeof (tensor_dim));
      config->rate_d = configs->rate_d;
      config->rate_n = configs->rate_n;
      ret = TRUE;
    }
      break;
    default:
      ret = FALSE;
  }

  return ret;
}

/**
 * @brief Looping to generete outbut buffer for srcpad
 * @param tensor_merge tensor merger
 * @param tensor_buf output buffer for srcpad
 * @param pts_time earliest pts time (present timestamp)
 * @param dts_time earliest dts time (decoding timestamp)
 * @return isEOS boolean EOS ( End of Stream )
 */
static gboolean
gst_tensor_merge_collect_buffer (GstTensorMerge * tensor_merge,
    GstBuffer * tensors_buf, GstClockTime * pts_time, GstClockTime * dts_time)
{
  GSList *walk = NULL;
  GstTensorMergePadData *bestpad = NULL;
  GstMemory *mem;
  gboolean isEOS = FALSE;
  gint old_numerator = G_MAXINT;
  gint old_denominator = G_MAXINT;
  gint counting = 0;
  GstTensorConfig config;

  walk = tensor_merge->collect->data;

  while (walk) {
    GstCollectData *data = (GstCollectData *) walk->data;
    GstTensorMergePadData *pad = (GstTensorMergePadData *) data;
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
    buf = gst_collect_pads_pop (tensor_merge->collect, data);

    if (GST_IS_BUFFER (buf)) {
      if (GST_BUFFER_PTS_IS_VALID (buf)) {
        if (data->segment.format == GST_FORMAT_TIME)
          pad->pts_timestamp =
              gst_segment_to_running_time (&data->segment, GST_FORMAT_TIME,
              GST_BUFFER_PTS (buf));
      } else {
        pad->pts_timestamp = GST_CLOCK_TIME_NONE;
      }
      if (buf && GST_BUFFER_DTS_IS_VALID (buf)) {
        if (data->segment.format == GST_FORMAT_TIME)
          pad->dts_timestamp =
              gst_segment_to_running_time (&data->segment, GST_FORMAT_TIME,
              GST_BUFFER_DTS (buf));
      } else {
        pad->dts_timestamp = GST_CLOCK_TIME_NONE;
      }

      mem = gst_buffer_get_memory (buf, 0);
      gst_buffer_append_memory (tensors_buf, mem);

      if (gst_tensor_merge_compare_pads (tensor_merge, bestpad, pad) > 0) {
        bestpad = pad;
        *pts_time = bestpad->pts_timestamp;
        *dts_time = bestpad->dts_timestamp;
      }

      gst_buffer_unref (buf);
    } else {
      isEOS = TRUE;
    }

    tensor_merge->tensors_config.info.info[counting] = config.info;
    counting++;
  }

  tensor_merge->tensors_config.rate_d = old_denominator;
  tensor_merge->tensors_config.rate_n = old_numerator;

  debug_print (!tensor_merge->silent, "pts %" GST_TIME_FORMAT,
      GST_TIME_ARGS (*pts_time));
  debug_print (!tensor_merge->silent, "dts %" GST_TIME_FORMAT,
      GST_TIME_ARGS (*dts_time));

  /* set timestamp */
  GST_BUFFER_PTS (tensors_buf) = *pts_time;
  GST_BUFFER_DTS (tensors_buf) = *dts_time;
  return isEOS;
}


/**
 * @brief Generate Output GstMemory
 * @param tensor_merge tensor merger
 * @param tensors_buf collected tensors buffer
 * @param tensor_buf output tensor buffer
 * @return boolean
 */
static GstFlowReturn
gst_tensor_merge_generate_mem (GstTensorMerge * tensor_merge,
    GstBuffer * tensors_buf, GstBuffer * tensor_buf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstMapInfo mInfo[NNS_TENSOR_SIZE_LIMIT];
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo outInfo;
  GstMemory *outMem;
  uint8_t *inptr, *outptr;
  int num_mem = tensor_merge->tensors_config.info.num_tensors;
  int i, j, k, l;
  size_t outSize = 0;
  tensor_dim dim;
  tensor_type type;

  memcpy (&dim, &tensor_merge->tensors_config.info.info[0].dimension,
      sizeof (tensor_dim));
  type = tensor_merge->tensors_config.info.info[0].type;

  for (i = 0; i < num_mem; i++) {
    mem[i] = gst_buffer_peek_memory (tensors_buf, i);
    g_assert (gst_memory_map (mem[i], &mInfo[i], GST_MAP_READ));
    outSize += mInfo[i].size;
  }

  outMem = gst_allocator_alloc (NULL, outSize, NULL);
  g_assert (gst_memory_map (outMem, &outInfo, GST_MAP_WRITE));
  outptr = outInfo.data;

  switch (tensor_merge->mode) {
    case GTT_LINEAR:
    {
      switch (tensor_merge->data_linear.direction) {
        case LINEAR_FIRST:
        {
          for (l = 0; l < dim[3]; l++) {
            for (i = 0; i < dim[2]; i++) {
              for (j = 0; j < dim[1]; j++) {
                for (k = 0; k < num_mem; k++) {
                  size_t c =
                      tensor_merge->tensors_config.info.info[k].dimension[0];
                  size_t s = tensor_element_size[type] * c;
                  inptr =
                      mInfo[k].data + (l * dim[2] * dim[1] + i * dim[1] +
                      j) * s;
                  memcpy (outptr, inptr, s);
                  outptr += s;
                }
              }
            }

          }
          ret = TRUE;
        }
          break;
        case LINEAR_SECOND:
        {
          for (l = 0; l < dim[3]; l++) {
            for (i = 0; i < dim[2]; i++) {
              for (k = 0; k < num_mem; k++) {
                size_t c = 1;
                for (j = 0; j < LINEAR_SECOND + 1; j++)
                  c *= tensor_merge->tensors_config.info.info[k].dimension[j];

                size_t s = tensor_element_size[type] * c;

                inptr = mInfo[k].data + (l * dim[2] + i) * s;

                memcpy (outptr, inptr, s);
                outptr += s;
              }
            }
          }
          ret = TRUE;
        }
          break;
        case LINEAR_THIRD:
        {
          for (l = 0; l < dim[3]; l++) {
            for (k = 0; k < num_mem; k++) {
              size_t c = 1;
              for (j = 0; j < LINEAR_THIRD + 1; j++)
                c *= tensor_merge->tensors_config.info.info[k].dimension[j];

              size_t s = tensor_element_size[type] * c;

              inptr = mInfo[k].data + l * s;

              memcpy (outptr, inptr, s);
              outptr += s;
            }
          }
          ret = TRUE;
        }
          break;
        case LINEAR_FOURTH:
        {
          for (k = 0; k < num_mem; k++) {
            size_t c = 1;
            for (j = 0; j < LINEAR_FOURTH + 1; j++)
              c *= tensor_merge->tensors_config.info.info[k].dimension[j];

            size_t s = tensor_element_size[type] * c;

            inptr = mInfo[k].data;
            memcpy (outptr, inptr, s);
            outptr += s;
          }

          ret = TRUE;
        }
          break;
        default:
          ret = FALSE;
      }
    }
      break;
    default:
      ret = FALSE;
  }

  gst_buffer_append_memory (tensor_buf, outMem);
  gst_buffer_copy_into (tensor_buf, tensors_buf, GST_BUFFER_COPY_TIMESTAMPS, 0,
      -1);
  for (i = 0; i < num_mem; i++)
    gst_memory_unmap (mem[i], &mInfo[i]);
  gst_memory_unmap (outMem, &outInfo);

  return ret;
}

/**
 * @brief Gst Collect Pads Function which is called once collect pads done.
 * @param pads GstCollectPads
 * @param tensor_merge Merger
 * @return GstFlowReturn
 */
static GstFlowReturn
gst_tensor_merge_collected (GstCollectPads * pads,
    GstTensorMerge * tensor_merge)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *tensors_buf, *tensor_buf;
  GstClockTime pts_time = GST_CLOCK_TIME_NONE;
  GstClockTime dts_time = GST_CLOCK_TIME_NONE;
  GstClockTime time = 0;
  gboolean isEOS = FALSE;
  GST_DEBUG_OBJECT (tensor_merge, " all pads are collected ");
  if (tensor_merge->need_stream_start) {
    gchar s_id[32];
    g_snprintf (s_id, sizeof (s_id), " tensormerge - %08x ", g_random_int ());
    gst_pad_push_event (tensor_merge->srcpad,
        gst_event_new_stream_start (s_id));
    tensor_merge->need_stream_start = FALSE;
  }

  tensors_buf = gst_buffer_new ();
  g_assert (tensors_buf != NULL);

  isEOS =
      gst_tensor_merge_collect_buffer (tensor_merge, tensors_buf, &pts_time,
      &dts_time);

  if (isEOS) {
    gst_pad_push_event (tensor_merge->srcpad, gst_event_new_eos ());
    ret = GST_FLOW_EOS;
    goto beach;
  }

  if (!tensor_merge->negotiated) {
    GstCaps *newcaps;
    GstTensorConfig config;

    if (gst_merge_tensors_config (tensor_merge, &tensor_merge->tensors_config,
            &config)) {
      g_assert (gst_tensor_config_validate (&config));
      newcaps = gst_tensor_caps_from_config (&config);
    } else {
      goto nego_error;
    }

    if (!gst_pad_set_caps (tensor_merge->srcpad, newcaps)) {
      gst_caps_unref (newcaps);
      goto nego_error;
    }

    gst_caps_unref (newcaps);
    tensor_merge->negotiated = TRUE;
  }

  if (tensor_merge->need_segment) {
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
    gst_pad_push_event (tensor_merge->srcpad, gst_event_new_segment (&segment));
    tensor_merge->need_segment = FALSE;
  }

  tensor_buf = gst_buffer_new ();
  g_assert (tensor_buf != NULL);

  gst_tensor_merge_generate_mem (tensor_merge, tensors_buf, tensor_buf);

  ret = gst_pad_push (tensor_merge->srcpad, tensor_buf);
  if (ret != GST_FLOW_OK) {
    GST_WARNING_OBJECT (tensor_merge, "pushed outbuf, result = %s",
        gst_flow_get_name (ret));
  }
beach:
  gst_buffer_unref (tensors_buf);
  return ret;
nego_error:
  {
    gst_buffer_unref (tensors_buf);
    GST_WARNING_OBJECT (tensor_merge, "failed to set caps");
    GST_ELEMENT_ERROR (tensor_merge, CORE, NEGOTIATION, (NULL), (NULL));
    return GST_FLOW_NOT_NEGOTIATED;
  }
}

/**
 * @brief Ready --> Pasuse State Change
 */
static void
gst_tensor_merge_ready_to_paused (GstTensorMerge * tensor_merge)
{
  tensor_merge->need_stream_start = TRUE;
  tensor_merge->need_segment = TRUE;
  tensor_merge->negotiated = FALSE;
  gst_collect_pads_start (tensor_merge->collect);
}

/**
 * @brief change state (gst element vmethod)
 */
static GstStateChangeReturn
gst_tensor_merge_change_state (GstElement * element, GstStateChange transition)
{
  GstTensorMerge *tensor_merge;
  GstStateChangeReturn ret;
  tensor_merge = GST_TENSOR_MERGE (element);
  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_tensor_merge_ready_to_paused (tensor_merge);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_collect_pads_stop (tensor_merge->collect);
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
 * @brief Setup internal data (data_* in GstTensor_Merge)
 * @param[in/out] filter "this" pointer. mode & option MUST BE set already.
 */
static void
gst_tensor_merge_set_option_data (GstTensorMerge * filter)
{
  if (filter->mode == GTT_END || filter->option == NULL)
    return;
  switch (filter->mode) {
    case GTT_LINEAR:
    {
      filter->data_linear.direction =
          find_key_strv (gst_tensor_merge_linear_string, filter->option);
      g_assert (filter->data_linear.direction >= 0);
      filter->loaded = TRUE;
    }
      break;
    default:
      g_printerr ("Cannot identify mode\n");
      g_assert (0);
  }
}

/**
 * @brief Get property (gst element vmethod)
 */
static void
gst_tensor_merge_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorMerge *filter = GST_TENSOR_MERGE (object);
  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    case PROP_MODE:
      filter->mode = gst_tensor_merge_get_mode (g_value_get_string (value));
      g_assert (filter->mode != GTT_END);
      gst_tensor_merge_set_option_data (filter);
      break;
    case PROP_OPTION:
      filter->option = g_value_dup_string (value);
      gst_tensor_merge_set_option_data (filter);
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
gst_tensor_merge_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorMerge *filter = GST_TENSOR_MERGE (object);
  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    case PROP_MODE:
      g_value_set_string (value, gst_tensor_merge_mode_string[filter->mode]);
      break;
    case PROP_OPTION:
      g_value_set_string (value, filter->option);
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
NNSTREAMER_PLUGIN_INIT (tensor_merge)
{
  /** debug category for fltering log messages
   * exchange the string 'Template tensor_merge' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_merge_debug, "tensor_merge", 0,
      "Tensor Merger");
  return gst_element_register (plugin, "tensor_merge",
      GST_RANK_NONE, GST_TYPE_TENSOR_MERGE);
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
 * @brief gstreamer looks for this structure to register tensormerge
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    tensor_merge,
    "tensor merge plugin",
    gst_tensor_merge_plugin_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer/");
#endif
