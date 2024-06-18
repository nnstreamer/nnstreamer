/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file    tensor_query_serversrc.c
 * @date    09 Jul 2021
 * @brief   GStreamer plugin to handle tensor query_server src
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <tensor_typedef.h>
#include <tensor_common.h>
#include "tensor_query_serversrc.h"
#include "tensor_query_common.h"
#include "nnstreamer_util.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversrc_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversrc_debug

#define DEFAULT_PORT_SRC 3000
#define DEFAULT_IS_LIVE TRUE
#define DEFAULT_MQTT_HOST "127.0.0.1"
#define DEFAULT_MQTT_PORT 1883
#define DEFAULT_DATA_POP_TIMEOUT 100000U

/**
 * @brief the capabilities of the outputs
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

/**
 * @brief query_serversrc properties
 */
enum
{
  PROP_0,
  PROP_HOST,
  PROP_PORT,
  PROP_DEST_HOST,
  PROP_DEST_PORT,
  PROP_CONNECT_TYPE,
  PROP_TIMEOUT,
  PROP_TOPIC,
  PROP_ID,
  PROP_IS_LIVE
};

#define gst_tensor_query_serversrc_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSrc, gst_tensor_query_serversrc,
    GST_TYPE_PUSH_SRC);

static GstStateChangeReturn gst_tensor_query_serversrc_change_state (GstElement
    * element, GstStateChange transition);
static void gst_tensor_query_serversrc_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_finalize (GObject * object);
static GstFlowReturn gst_tensor_query_serversrc_create (GstPushSrc * psrc,
    GstBuffer ** buf);
static gboolean gst_tensor_query_serversrc_set_caps (GstBaseSrc * bsrc,
    GstCaps * caps);

/**
 * @brief initialize the query_serversrc class
 */
static void
gst_tensor_query_serversrc_class_init (GstTensorQueryServerSrcClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *gstbasesrc_class;
  GstPushSrcClass *gstpushsrc_class;

  gstpushsrc_class = (GstPushSrcClass *) klass;
  gstbasesrc_class = (GstBaseSrcClass *) gstpushsrc_class;
  gstelement_class = (GstElementClass *) gstbasesrc_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_query_serversrc_set_property;
  gobject_class->get_property = gst_tensor_query_serversrc_get_property;
  gobject_class->finalize = gst_tensor_query_serversrc_finalize;
  gstelement_class->change_state = gst_tensor_query_serversrc_change_state;
  gstpushsrc_class->create = gst_tensor_query_serversrc_create;

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host", "The hostname to listen as",
          DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "The port to listen to (0=random available port)", 0,
          65535, DEFAULT_PORT_SRC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_DEST_HOST,
      g_param_spec_string ("dest-host", "Destination Host",
          "The destination hostname to connect", DEFAULT_MQTT_HOST,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_DEST_PORT,
      g_param_spec_uint ("dest-port", "Destination Port",
          "The destination port to connect to (0=random available port)", 0,
          65535, DEFAULT_MQTT_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_CONNECT_TYPE,
      g_param_spec_enum ("connect-type", "Connect Type", "The connection type.",
          GST_TYPE_QUERY_CONNECT_TYPE, DEFAULT_CONNECT_TYPE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TIMEOUT,
      g_param_spec_uint ("timeout", "Timeout",
          "The timeout as seconds to maintain connection", 0, 3600,
          QUERY_DEFAULT_TIMEOUT_SEC,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TOPIC,
      g_param_spec_string ("topic", "Topic",
          "The main topic of the host and option if necessary. "
          "(topic)/(optional topic for main topic).", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_ID,
      g_param_spec_uint ("id", "ID", "ID for distinguishing query servers.", 0,
          G_MAXUINT, DEFAULT_SERVER_ID,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_IS_LIVE,
      g_param_spec_boolean ("is-live", "Is Live",
          "Synchronize the incoming buffers' timestamp with the current running time",
          DEFAULT_IS_LIVE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorQueryServerSrc", "Source/Tensor/Query",
      "Receive tensor data as a server over the network",
      "Samsung Electronics Co., Ltd.");

  GST_DEBUG_CATEGORY_INIT (gst_tensor_query_serversrc_debug,
      "tensor_query_serversrc", 0, "Tensor Query Server Source");
}

/**
 * @brief initialize the new query_serversrc element
 */
static void
gst_tensor_query_serversrc_init (GstTensorQueryServerSrc * src)
{
  src->host = g_strdup (DEFAULT_HOST);
  src->port = DEFAULT_PORT_SRC;
  src->dest_host = g_strdup (DEFAULT_MQTT_HOST);
  src->dest_port = DEFAULT_MQTT_PORT;
  src->connect_type = DEFAULT_CONNECT_TYPE;
  src->timeout = QUERY_DEFAULT_TIMEOUT_SEC;
  src->topic = NULL;
  src->src_id = DEFAULT_SERVER_ID;
  src->configured = FALSE;
  src->msg_queue = g_async_queue_new ();
  src->playing = FALSE;

  gst_base_src_set_format (GST_BASE_SRC (src), GST_FORMAT_TIME);
  /** set the timestamps on each buffer */
  gst_base_src_set_do_timestamp (GST_BASE_SRC (src), TRUE);
  /** set the source to be live */
  gst_base_src_set_live (GST_BASE_SRC (src), DEFAULT_IS_LIVE);
}

/**
 * @brief finalize the query_serversrc object
 */
static void
gst_tensor_query_serversrc_finalize (GObject * object)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (object);
  nns_edge_data_h data_h;

  g_free (src->host);
  src->host = NULL;
  g_free (src->dest_host);
  src->dest_host = NULL;
  g_free (src->topic);
  src->topic = NULL;

  while ((data_h = g_async_queue_try_pop (src->msg_queue))) {
    nns_edge_data_destroy (data_h);
  }
  g_async_queue_unref (src->msg_queue);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief nnstreamer-edge event callback.
 */
static int
_nns_edge_event_cb (nns_edge_event_h event_h, void *user_data)
{
  nns_edge_event_e event_type;
  int ret = NNS_EDGE_ERROR_NONE;

  GstTensorQueryServerSrc *src = (GstTensorQueryServerSrc *) user_data;
  ret = nns_edge_event_get_type (event_h, &event_type);
  if (NNS_EDGE_ERROR_NONE != ret) {
    nns_loge ("Failed to get event type!");
    return ret;
  }

  switch (event_type) {
    case NNS_EDGE_EVENT_NEW_DATA_RECEIVED:
    {
      nns_edge_data_h data;

      ret = nns_edge_event_parse_new_data (event_h, &data);
      if (NNS_EDGE_ERROR_NONE != ret) {
        nns_loge ("Failed to parse new data received from new data event");
        return ret;
      }
      g_async_queue_push (src->msg_queue, data);
      break;
    }
    default:
      break;
  }

  return ret;
}

/**
 * @brief start processing of query_serversrc, setting up the server
 */
static gboolean
_gst_tensor_query_serversrc_start (GstTensorQueryServerSrc * src)
{
  gboolean ret = FALSE;

  if (gst_tensor_query_server_add_data (src->src_id))
    ret = gst_tensor_query_server_wait_sink (src->src_id);

  if (!ret)
    nns_loge ("Failed to get server information from query server.");

  return ret;
}

/**
 * @brief start processing of query_serversrc, setting up the server
 */
static gboolean
_gst_tensor_query_serversrc_playing (GstTensorQueryServerSrc * src,
    nns_edge_connect_type_e connect_type)
{
  GstTensorQueryEdgeInfo edge_info = { 0 };
  gboolean ret;

  edge_info.host = src->host;
  edge_info.port = src->port;
  edge_info.dest_host = src->dest_host;
  edge_info.dest_port = src->dest_port;
  edge_info.topic = src->topic;
  edge_info.cb = _nns_edge_event_cb;
  edge_info.pdata = src;

  ret = gst_tensor_query_server_prepare (src->src_id, connect_type, &edge_info);

  return ret;
}

/**
 * @brief Change state of query server src.
 */
static GstStateChangeReturn
gst_tensor_query_serversrc_change_state (GstElement * element,
    GstStateChange transition)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (element);
  GstBaseSrc *bsrc = GST_BASE_SRC (element);
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstCaps *caps;

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      if (!_gst_tensor_query_serversrc_playing (src, src->connect_type)) {
        nns_loge ("Failed to change state from PAUSED to PLAYING.");
        return GST_STATE_CHANGE_FAILURE;
      }

      caps = gst_pad_peer_query_caps (GST_BASE_SRC_PAD (bsrc), NULL);
      gst_tensor_query_serversrc_set_caps (bsrc, caps);
      gst_caps_unref (caps);

      src->playing = TRUE;
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      if (!_gst_tensor_query_serversrc_start (src)) {
        nns_loge ("Failed to change state from READY to PAUSED.");
        return GST_STATE_CHANGE_FAILURE;
      }
      break;
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
      src->playing = FALSE;
      gst_tensor_query_server_release_edge_handle (src->src_id);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_tensor_query_server_remove_data (src->src_id);
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief set property of query_serversrc
 */
static void
gst_tensor_query_serversrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSrc *serversrc = GST_TENSOR_QUERY_SERVERSRC (object);

  switch (prop_id) {
    case PROP_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("host property cannot be NULL");
        break;
      }
      g_free (serversrc->host);
      serversrc->host = g_value_dup_string (value);
      break;
    case PROP_PORT:
      serversrc->port = g_value_get_uint (value);
      break;
    case PROP_DEST_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("host property cannot be NULL");
        break;
      }
      g_free (serversrc->dest_host);
      serversrc->dest_host = g_value_dup_string (value);
      break;
    case PROP_DEST_PORT:
      serversrc->dest_port = g_value_get_uint (value);
      break;
    case PROP_CONNECT_TYPE:
      serversrc->connect_type = g_value_get_enum (value);
      break;
    case PROP_TIMEOUT:
      serversrc->timeout = g_value_get_uint (value);
      break;
    case PROP_TOPIC:
      if (!g_value_get_string (value)) {
        nns_logw ("topic property cannot be NULL. Query-hybrid is disabled.");
        break;
      }
      g_free (serversrc->topic);
      serversrc->topic = g_value_dup_string (value);
      break;
    case PROP_ID:
      serversrc->src_id = g_value_get_uint (value);
      break;
    case PROP_IS_LIVE:
      gst_base_src_set_live (GST_BASE_SRC (serversrc),
          g_value_get_boolean (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property of query_serversrc
 */
static void
gst_tensor_query_serversrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorQueryServerSrc *serversrc = GST_TENSOR_QUERY_SERVERSRC (object);

  switch (prop_id) {
    case PROP_HOST:
      g_value_set_string (value, serversrc->host);
      break;
    case PROP_PORT:
      g_value_set_uint (value, serversrc->port);
      break;
    case PROP_DEST_HOST:
      g_value_set_string (value, serversrc->dest_host);
      break;
    case PROP_DEST_PORT:
      g_value_set_uint (value, serversrc->dest_port);
      break;
    case PROP_CONNECT_TYPE:
      g_value_set_enum (value, serversrc->connect_type);
      break;
    case PROP_TIMEOUT:
      g_value_set_uint (value, serversrc->timeout);
      break;
    case PROP_TOPIC:
      g_value_set_string (value, serversrc->topic);
      break;
    case PROP_ID:
      g_value_set_uint (value, serversrc->src_id);
      break;
    case PROP_IS_LIVE:
      g_value_set_boolean (value,
          gst_base_src_is_live (GST_BASE_SRC (serversrc)));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Get buffer from message queue.
 */
static GstBuffer *
_gst_tensor_query_serversrc_get_buffer (GstTensorQueryServerSrc * src)
{
  nns_edge_data_h data_h = NULL;
  GstBuffer *buffer = NULL;
  guint i, num_data;
  GstMetaQuery *meta_query;
  int ret;

  while (src->playing && !data_h) {
    data_h = g_async_queue_timeout_pop (src->msg_queue,
        DEFAULT_DATA_POP_TIMEOUT);
  }

  if (!data_h) {
    nns_loge ("Failed to get message from the server message queue.");
    return NULL;
  }

  ret = nns_edge_data_get_count (data_h, &num_data);
  if (ret != NNS_EDGE_ERROR_NONE || num_data == 0) {
    nns_loge ("Failed to get the number of memories of the edge data.");
    goto done;
  }

  buffer = gst_buffer_new ();
  for (i = 0; i < num_data; i++) {
    void *data = NULL;
    nns_size_t data_len = 0;
    gpointer new_data;

    nns_edge_data_get (data_h, i, &data, &data_len);
    new_data = _g_memdup (data, data_len);

    gst_buffer_append_memory (buffer,
        gst_memory_new_wrapped (0, new_data, data_len, 0, data_len, new_data,
            g_free));
  }

  meta_query = gst_buffer_add_meta_query (buffer);
  if (meta_query) {
    char *val;

    ret = nns_edge_data_get_info (data_h, "client_id", &val);
    if (NNS_EDGE_ERROR_NONE != ret) {
      gst_buffer_unref (buffer);
      buffer = NULL;
    } else {
      meta_query->client_id = g_ascii_strtoll (val, NULL, 10);
      g_free (val);
    }
  }

done:
  nns_edge_data_destroy (data_h);
  return buffer;
}

/**
 * @brief create query_serversrc, wait on socket and receive data
 */
static GstFlowReturn
gst_tensor_query_serversrc_create (GstPushSrc * psrc, GstBuffer ** outbuf)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (psrc);
  GstBaseSrc *bsrc = GST_BASE_SRC (psrc);
  GstStateChangeReturn sret;
  GstState state = GST_STATE_NULL;

  if (!src->configured) {
    GstCaps *caps = gst_pad_peer_query_caps (GST_BASE_SRC_PAD (bsrc), NULL);
    if (gst_caps_is_fixed (caps)) {
      gst_base_src_set_caps (bsrc, caps);
    }

    gst_tensor_query_serversrc_set_caps (bsrc, caps);

    gst_caps_unref (caps);
    src->configured = TRUE;
  }

  *outbuf = _gst_tensor_query_serversrc_get_buffer (src);
  if (*outbuf == NULL) {
    sret = gst_element_get_state (GST_ELEMENT (psrc), &state, NULL, 0);
    if (sret != GST_STATE_CHANGE_SUCCESS || state != GST_STATE_PLAYING) {
      nns_logw ("Failed to get buffer for query server, not in PLAYING state.");
      return GST_FLOW_FLUSHING;
    }

    nns_loge ("Failed to get buffer to push to the tensor query serversrc.");
    return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * @brief An implementation of the set_caps vmethod in GstBaseSrcClass
 */
static gboolean
gst_tensor_query_serversrc_set_caps (GstBaseSrc * bsrc, GstCaps * caps)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);
  gchar *caps_str, *new_caps_str;

  caps_str = gst_caps_to_string (caps);

  new_caps_str = g_strdup_printf ("@query_server_src_caps@%s", caps_str);
  gst_tensor_query_server_set_caps (src->src_id, new_caps_str);

  g_free (new_caps_str);
  g_free (caps_str);

  return TRUE;
}
