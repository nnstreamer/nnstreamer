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
#include "tensor_query_common.h"
#include "tensor_query_serversrc.h"
#include "tensor_query_server.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_query_serversrc_debug);
#define GST_CAT_DEFAULT gst_tensor_query_serversrc_debug

#define DEFAULT_HOST "localhost"
#define DEFAULT_PORT_SRC 3001
#define DEFAULT_PROTOCOL _TENSOR_QUERY_PROTOCOL_TCP
#define DEFAULT_IS_LIVE TRUE
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
  PROP_PROTOCOL,
  PROP_TIMEOUT,
  PROP_OPERATION,
  PROP_BROKER_HOST,
  PROP_BROKER_PORT,
  PROP_ID,
  PROP_IS_LIVE
};

#define gst_tensor_query_serversrc_parent_class parent_class
G_DEFINE_TYPE (GstTensorQueryServerSrc, gst_tensor_query_serversrc,
    GST_TYPE_PUSH_SRC);

static void gst_tensor_query_serversrc_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_tensor_query_serversrc_finalize (GObject * object);

static gboolean gst_tensor_query_serversrc_start (GstBaseSrc * bsrc);
static gboolean gst_tensor_query_serversrc_stop (GstBaseSrc * bsrc);
static GstFlowReturn gst_tensor_query_serversrc_create (GstPushSrc * psrc,
    GstBuffer ** buf);

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

  g_object_class_install_property (gobject_class, PROP_HOST,
      g_param_spec_string ("host", "Host", "The hostname to listen as",
          DEFAULT_HOST, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PORT,
      g_param_spec_uint ("port", "Port",
          "The port to listen to (0=random available port)", 0,
          65535, DEFAULT_PORT_SRC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PROTOCOL,
      g_param_spec_int ("protocol", "Protocol",
          "The network protocol to establish connection", 0,
          _TENSOR_QUERY_PROTOCOL_END, DEFAULT_PROTOCOL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TIMEOUT,
      g_param_spec_uint ("timeout", "Timeout",
          "The timeout as seconds to maintain connection", 0,
          3600, QUERY_DEFAULT_TIMEOUT_SEC,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OPERATION,
      g_param_spec_string ("operation", "Operation",
          "The main operation of the host and option if necessary. "
          "(operation)/(optional topic for main operation).",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_BROKER_HOST,
      g_param_spec_string ("broker-host", "Broker Host",
          "Broker host address to connect.", DEFAULT_BROKER_HOST,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_BROKER_PORT,
      g_param_spec_uint ("broker-port", "Broker Port",
          "Broker port to connect.", 0, 65535,
          DEFAULT_BROKER_PORT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_ID,
      g_param_spec_uint ("id", "ID",
          "ID for distinguishing query servers.", 0,
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

  gstbasesrc_class->start = gst_tensor_query_serversrc_start;
  gstbasesrc_class->stop = gst_tensor_query_serversrc_stop;
  gstpushsrc_class->create = gst_tensor_query_serversrc_create;

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
  src->protocol = DEFAULT_PROTOCOL;
  src->timeout = QUERY_DEFAULT_TIMEOUT_SEC;
  src->operation = NULL;
  src->broker_host = g_strdup (DEFAULT_BROKER_HOST);
  src->broker_port = DEFAULT_BROKER_PORT;
  src->src_id = DEFAULT_SERVER_ID;
  tensor_query_hybrid_init (&src->hybrid_info, NULL, 0, TRUE);
  src->server_data = nnstreamer_query_server_data_new ();
  src->configured = FALSE;
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

  g_free (src->host);
  src->host = NULL;
  g_free (src->operation);
  src->operation = NULL;
  g_free (src->broker_host);
  src->broker_host = NULL;
  if (src->server_data) {
    nnstreamer_query_server_data_free (src->server_data);
    src->server_data = NULL;
  }
  G_OBJECT_CLASS (parent_class)->finalize (object);
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
    case PROP_PROTOCOL:
      serversrc->protocol = g_value_get_int (value);
      break;
    case PROP_TIMEOUT:
      serversrc->timeout = g_value_get_uint (value);
      break;
    case PROP_OPERATION:
      if (!g_value_get_string (value)) {
        nns_logw
            ("operation property cannot be NULL. Query-hybrid is disabled.");
        break;
      }
      g_free (serversrc->operation);
      serversrc->operation = g_value_dup_string (value);
      break;
    case PROP_BROKER_HOST:
      if (!g_value_get_string (value)) {
        nns_logw ("Broker host property cannot be NULL");
        break;
      }
      g_free (serversrc->broker_host);
      serversrc->broker_host = g_value_dup_string (value);
      break;
    case PROP_BROKER_PORT:
      serversrc->broker_port = g_value_get_uint (value);
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
    case PROP_PROTOCOL:
      g_value_set_int (value, serversrc->protocol);
      break;
    case PROP_TIMEOUT:
      g_value_set_uint (value, serversrc->timeout);
      break;
    case PROP_OPERATION:
      g_value_set_string (value, serversrc->operation);
      break;
    case PROP_BROKER_HOST:
      g_value_set_string (value, serversrc->broker_host);
      break;
    case PROP_BROKER_PORT:
      g_value_set_uint (value, serversrc->broker_port);
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
 * @brief start processing of query_serversrc, setting up the server
 */
static gboolean
gst_tensor_query_serversrc_start (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);

  if (!src->server_data) {
    nns_loge ("Server_data is NULL");
    return FALSE;
  }

  if (nnstreamer_query_server_init (src->server_data, src->protocol,
          src->host, src->port, TRUE) != 0) {
    nns_loge ("Failed to setup server");
    return FALSE;
  }

  src->server_info_h = gst_tensor_query_server_add_data (src->src_id);
  if (!gst_tensor_query_server_wait_sink (src->server_info_h)) {
    nns_loge ("Failed to get server information from query server.");
    return FALSE;
  }

  /** Publish query sever connection info */
  if (src->operation) {
    tensor_query_hybrid_set_node (&src->hybrid_info, src->host, src->port,
        src->server_info_h);
    tensor_query_hybrid_set_broker (&src->hybrid_info,
        src->broker_host, src->broker_port);

    if (!tensor_query_hybrid_publish (&src->hybrid_info, src->operation)) {
      nns_loge ("Failed to publish a topic.");
      return FALSE;
    }
  } else {
    nns_logi ("Query-hybrid feature is disabled.");
    nns_logi
        ("Specify operation to register server to broker (e.g., operation=object_detection/mobilev3).");
  }
  return TRUE;
}

/**
 * @brief stop processing of query_serversrc
 */
static gboolean
gst_tensor_query_serversrc_stop (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);

  tensor_query_hybrid_close (&src->hybrid_info);
  nnstreamer_query_server_data_free (src->server_data);
  src->server_data = NULL;
  return TRUE;
}

/**
 * @brief create query_serversrc, wait on socket and receive data
 */
static GstFlowReturn
gst_tensor_query_serversrc_create (GstPushSrc * psrc, GstBuffer ** outbuf)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (psrc);
  GstBaseSrc *bsrc = GST_BASE_SRC (psrc);

  if (!src->server_data) {
    nns_loge ("Server data is NULL");
    return GST_FLOW_ERROR;
  }

  if (!src->configured) {
    gchar *sink_caps_str, *src_caps_str;

    GstCaps *caps = gst_pad_peer_query_caps (GST_BASE_SRC_PAD (bsrc), NULL);
    if (gst_caps_is_fixed (caps)) {
      gst_base_src_set_caps (bsrc, caps);
    }

    src->server_info_h = gst_tensor_query_server_get_data (src->src_id);
    sink_caps_str =
        gst_tensor_query_server_get_sink_caps_str (src->server_info_h);
    src_caps_str = gst_caps_to_string (caps);
    nnstreamer_query_server_data_set_caps_str (src->server_data, src_caps_str,
        sink_caps_str);

    g_free (sink_caps_str);
    g_free (src_caps_str);
    gst_caps_unref (caps);
    src->configured = TRUE;
  }

  *outbuf = nnstreamer_query_server_get_buffer (src->server_data);

  return GST_FLOW_OK;
}
