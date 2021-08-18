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
#define DEFAULT_TIMEOUT 10

#define CAPS_STRING GST_TENSORS_CAP_DEFAULT

/**
 * @brief the capabilities of the outputs
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief query_serversrc properties
 */
enum
{
  PROP_0,
  PROP_HOST,
  PROP_PORT,
  PROP_PROTOCOL,
  PROP_TIMEOUT
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
      g_param_spec_int ("port", "Port",
          "The port to listen to (0=random available port)", 0,
          65535, DEFAULT_PORT_SRC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_PROTOCOL,
      g_param_spec_int ("protocol", "Protocol",
          "The network protocol to establish connection", 0,
          _TENSOR_QUERY_PROTOCOL_END, DEFAULT_PROTOCOL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_TIMEOUT,
      g_param_spec_int ("timeout", "Timeout",
          "The timeout as seconds to maintain connection", 0,
          3600, DEFAULT_TIMEOUT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  src->timeout = DEFAULT_TIMEOUT;
  gst_tensors_config_init (&src->src_config);
  src->server_data = nnstreamer_query_server_data_new ();
}

/**
 * @brief finalize the query_serversrc object
 */
static void
gst_tensor_query_serversrc_finalize (GObject * object)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (object);

  g_free (src->host);
  gst_tensors_config_free (&src->src_config);
  nnstreamer_query_server_data_free (src->server_data);
  src->server_data = NULL;
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
      serversrc->timeout = g_value_get_int (value);
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
      g_value_set_int (value, serversrc->port);
      break;
    case PROP_PROTOCOL:
      g_value_set_int (value, serversrc->protocol);
      break;
    case PROP_TIMEOUT:
      g_value_set_int (value, serversrc->timeout);
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

  gst_tensors_config_from_peer (bsrc->srcpad, &src->src_config, NULL);
  src->src_config.rate_n = 0;
  src->src_config.rate_d = 1;
  if (!gst_tensors_config_validate (&src->src_config)) {
    nns_loge ("Invalid tensors config from peer");
    return FALSE;
  }

  if (!src->server_data) {
    nns_loge ("Server_data is NULL");
    return FALSE;
  }

  if (nnstreamer_query_server_init (src->server_data, src->protocol,
          src->host, src->port) != 0) {
    nns_loge ("Failed to setup server");
    return FALSE;
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
  GstMemory *mem = NULL;
  GstMapInfo map;
  GstMetaQuery *meta_query;
  TensorQueryCommandData cmd_data;
  TensorQueryDataInfo data_info;
  query_connection_handle conn;
  guint i;
  gint ecode;

  if (!src->server_data) {
    nns_loge ("Server data is NULL");
    return GST_FLOW_ERROR;
  }

  while (TRUE) {
    conn = nnstreamer_query_server_accept (src->server_data);
    if (!conn) {
      nns_loge ("Failed to accept connection");
      goto error;
    }

    if (nnstreamer_query_receive (conn, &cmd_data, src->timeout) != 0) {
      nns_logi ("Failed to receive cmd");
      continue;
    }

    switch (cmd_data.cmd) {
      case _TENSOR_QUERY_CMD_REQUEST_INFO:
        if (gst_tensors_info_is_equal (&cmd_data.data_info.config.info,
                &src->src_config.info)) {
          cmd_data.cmd = _TENSOR_QUERY_CMD_RESPOND_APPROVE;
          /* respond sink config */
          gst_tensor_query_server_get_sink_config (&cmd_data.data_info.config);
        } else {
          /* respond deny with src config */
          nns_logw ("tensor info is not equal");
          cmd_data.cmd = _TENSOR_QUERY_CMD_RESPOND_DENY;
          gst_tensors_config_copy (&cmd_data.data_info.config,
              &src->src_config);
        }
        if (nnstreamer_query_send (conn, &cmd_data, src->timeout) != 0) {
          nns_logi ("Failed to send respond");
          continue;
        }
        break;

      case _TENSOR_QUERY_CMD_TRANSFER_START:
        data_info = cmd_data.data_info;
        *outbuf = gst_buffer_new ();
        for (i = 0; i < data_info.num_mems; i++) {
          mem = gst_allocator_alloc (NULL, data_info.mem_sizes[i], NULL);
          gst_buffer_append_memory (*outbuf, mem);

          if (!gst_memory_map (mem, &map, GST_MAP_READWRITE)) {
            nns_loge ("Failed to map the memory to receive data.");
            goto reset_buffer;
          }

          cmd_data.data.data = map.data;
          cmd_data.data.size = map.size;

          ecode = nnstreamer_query_receive (conn, &cmd_data, src->timeout);
          gst_memory_unmap (mem, &map);

          if (ecode != 0) {
            nns_logi ("Failed to receive data");
            goto reset_buffer;
          }
        }

        /* receive end */
        if (nnstreamer_query_receive (conn, &cmd_data, src->timeout) != 0 ||
            cmd_data.cmd != _TENSOR_QUERY_CMD_TRANSFER_END) {
          nns_logi ("Failed to receive end command");
          goto reset_buffer;
        }

        meta_query = gst_buffer_add_meta_query (*outbuf);
        if (meta_query) {
          meta_query->host =
              g_strdup (nnstreamer_query_connection_get_host (conn));
        }
        return GST_FLOW_OK;

      default:
        nns_loge ("Invalid cmd type %d", cmd_data.cmd);
        break;
    }
    continue;
  reset_buffer:
    gst_buffer_unref (*outbuf);
  }

error:
  gst_buffer_unref (*outbuf);
  return GST_FLOW_ERROR;
}
