/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversink.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query server sink
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_query_serversink.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversink_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversink_debug

#define DEFAULT_METALESS_FRAME_LIMIT 1

/**
 * @brief the capabilities of the inputs.
 */
static GstStaticPadTemplate sinktemplate = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

enum
{
  PROP_0,
  PROP_CONNECT_TYPE,
  PROP_ID,
  PROP_TIMEOUT,
  PROP_METALESS_FRAME_LIMIT
};

#define gst_tensor_query_serversink_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSink, gst_tensor_query_serversink,
    GST_TYPE_BASE_SINK);
static GstStateChangeReturn gst_tensor_query_serversink_change_state (GstElement
    * element, GstStateChange transition);
static void gst_tensor_query_serversink_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversink_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversink_finalize (GObject * object);

static GstFlowReturn gst_tensor_query_serversink_render (GstBaseSink * bsink,
    GstBuffer * buf);
static gboolean gst_tensor_query_serversink_set_caps (GstBaseSink * basesink,
    GstCaps * caps);

/**
 * @brief initialize the class
 */
static void
gst_tensor_query_serversink_class_init (GstTensorQueryServerSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gstbasesink_class = (GstBaseSinkClass *) klass;
  gstelement_class = (GstElementClass *) gstbasesink_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gstelement_class->change_state = gst_tensor_query_serversink_change_state;
  gobject_class->set_property = gst_tensor_query_serversink_set_property;
  gobject_class->get_property = gst_tensor_query_serversink_get_property;
  gobject_class->finalize = gst_tensor_query_serversink_finalize;

  g_object_class_install_property (gobject_class, PROP_CONNECT_TYPE,
      g_param_spec_enum ("connect-type", "Connect Type",
          "The connection type",
          GST_TYPE_QUERY_CONNECT_TYPE, DEFAULT_CONNECT_TYPE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TIMEOUT,
      g_param_spec_uint ("timeout", "Timeout",
          "The timeout as seconds to maintain connection", 0,
          3600, QUERY_DEFAULT_TIMEOUT_SEC,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_ID,
      g_param_spec_uint ("id", "ID",
          "ID for distinguishing query servers.", 0,
          G_MAXUINT, DEFAULT_SERVER_ID,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_METALESS_FRAME_LIMIT,
      g_param_spec_int ("limit", "Limit",
          "Limits of the number of the buffers that the server cannot handle. "
          "e.g., If the received buffer does not have a GstMetaQuery, the server cannot handle the buffer.",
          0, 65535, DEFAULT_METALESS_FRAME_LIMIT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sinktemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSink", "Sink/Tensor/Query",
      "Send tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  gstbasesink_class->set_caps = gst_tensor_query_serversink_set_caps;
  gstbasesink_class->render = gst_tensor_query_serversink_render;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_serversink_debug,
      "tensor_query_serversink", 0, "Tensor Query Server Sink");
}

/**
 * @brief initialize the new element
 */
static void
gst_tensor_query_serversink_init (GstTensorQueryServerSink * sink)
{
  sink->connect_type = DEFAULT_CONNECT_TYPE;
  sink->timeout = QUERY_DEFAULT_TIMEOUT_SEC;
  sink->sink_id = DEFAULT_SERVER_ID;
  sink->metaless_frame_count = 0;
}

/**
 * @brief finalize the object
 */
static void
gst_tensor_query_serversink_finalize (GObject * object)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (object);
  gst_tensor_query_server_remove_data (sink->server_h);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief start processing of query_serversink
 */
static void
_gst_tensor_query_serversink_start (GstTensorQueryServerSink * sink)
{
  gchar *id_str = NULL;

  id_str = g_strdup_printf ("%u", sink->sink_id);
  sink->server_h = gst_tensor_query_server_add_data (id_str);
  g_free (id_str);

  gst_tensor_query_server_set_configured (sink->server_h);
}

/**
 * @brief start processing of query_serversink
 */
static void
_gst_tensor_query_serversink_playing (GstTensorQueryServerSink * sink)
{
  gchar *id_str = NULL;

  id_str = g_strdup_printf ("%u", sink->sink_id);
  sink->edge_h =
      gst_tensor_query_server_get_edge_handle (id_str, sink->connect_type);
  g_free (id_str);
}

/**
 * @brief Change state of query server sink.
 */
static GstStateChangeReturn
gst_tensor_query_serversink_change_state (GstElement * element,
    GstStateChange transition)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (element);
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      _gst_tensor_query_serversink_playing (sink);
      if (!sink->edge_h) {
        nns_loge ("Failed to change state from PAUSED to PLAYING.");
        return GST_STATE_CHANGE_FAILURE;
      }
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      _gst_tensor_query_serversink_start (sink);
      if (!sink->server_h) {
        nns_loge ("Failed to change state from READY to PAUSED.");
        return GST_STATE_CHANGE_FAILURE;
      }
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    nns_loge ("Failed to change state");
    return ret;
  }

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      gst_tensor_query_server_release_edge_handle (sink->server_h);
      sink->edge_h = NULL;
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief set property
 */
static void
gst_tensor_query_serversink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSink *serversink = GST_TENSOR_QUERY_SERVERSINK (object);

  switch (prop_id) {
    case PROP_CONNECT_TYPE:
      serversink->connect_type = g_value_get_enum (value);
      break;
    case PROP_TIMEOUT:
      serversink->timeout = g_value_get_uint (value);
      break;
    case PROP_ID:
      serversink->sink_id = g_value_get_uint (value);
      break;
    case PROP_METALESS_FRAME_LIMIT:
      serversink->metaless_frame_limit = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property
 */
static void
gst_tensor_query_serversink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSink *serversink = GST_TENSOR_QUERY_SERVERSINK (object);

  switch (prop_id) {
    case PROP_CONNECT_TYPE:
      g_value_set_enum (value, serversink->connect_type);
      break;
    case PROP_TIMEOUT:
      g_value_set_uint (value, serversink->timeout);
      break;
    case PROP_ID:
      g_value_set_uint (value, serversink->sink_id);
      break;
    case PROP_METALESS_FRAME_LIMIT:
      g_value_set_int (value, serversink->metaless_frame_limit);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief An implementation of the set_caps vmethod in GstBaseSinkClass
 */
static gboolean
gst_tensor_query_serversink_set_caps (GstBaseSink * bsink, GstCaps * caps)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  gchar *caps_str, *new_caps_str;

  caps_str = gst_caps_to_string (caps);

  new_caps_str = g_strdup_printf ("@query_server_sink_caps@%s", caps_str);
  gst_tensor_query_server_set_caps (sink->server_h, new_caps_str);

  g_free (new_caps_str);
  g_free (caps_str);

  return TRUE;
}

/**
 * @brief render buffer, send buffer to client
 */
static GstFlowReturn
gst_tensor_query_serversink_render (GstBaseSink * bsink, GstBuffer * buf)
{
  GstTensorQueryServerSink *sink = GST_TENSOR_QUERY_SERVERSINK (bsink);
  GstMetaQuery *meta_query;
  nns_edge_data_h data_h;
  guint i, num_tensors = 0;
  gint ret;
  GstMemory *mem[NNS_TENSOR_SIZE_LIMIT];
  GstMapInfo map[NNS_TENSOR_SIZE_LIMIT];
  char *val;

  meta_query = gst_buffer_get_meta_query (buf);
  if (meta_query) {
    sink->metaless_frame_count = 0;

    ret = nns_edge_data_create (&data_h);
    if (ret != NNS_EDGE_ERROR_NONE) {
      nns_loge ("Failed to create data handle in server sink.");
      return GST_FLOW_ERROR;
    }

    num_tensors = gst_tensor_buffer_get_count (buf);
    for (i = 0; i < num_tensors; i++) {
      mem[i] = gst_tensor_buffer_get_nth_memory (buf, i);
      if (!gst_memory_map (mem[i], &map[i], GST_MAP_READ)) {
        ml_loge ("Cannot map the %uth memory in gst-buffer.", i);
        gst_memory_unref (mem[i]);
        num_tensors = i;
        nns_edge_data_destroy (data_h);
        goto done;
      }
      nns_edge_data_add (data_h, map[i].data, map[i].size, NULL);
    }

    val = g_strdup_printf ("%lld", (long long) meta_query->client_id);
    nns_edge_data_set_info (data_h, "client_id", val);
    g_free (val);

    nns_edge_send (sink->edge_h, data_h);
    nns_edge_data_destroy (data_h);
  } else {
    nns_logw ("Cannot get tensor query meta. Drop buffers!\n");
    sink->metaless_frame_count++;

    if (sink->metaless_frame_count >= sink->metaless_frame_limit) {
      nns_logw ("Cannot get tensor query meta. Stop the query server!\n"
          "There are elements that are not available on the query server.\n"
          "Please check available elements on the server."
          "See: https://github.com/nnstreamer/nnstreamer/wiki/Available-elements-on-query-server");
      return GST_FLOW_ERROR;
    }
  }
done:
  for (i = 0; i < num_tensors; i++) {
    gst_memory_unmap (mem[i], &map[i]);
    gst_memory_unref (mem[i]);
  }

  return GST_FLOW_OK;
}
