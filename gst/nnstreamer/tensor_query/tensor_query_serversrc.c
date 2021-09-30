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
#define DEFAULT_MQTT_HOST_ADDRESS "tcp://localhost"
#define DEFAULT_MQTT_HOST_PORT "1883"

#define CAPS_STRING GST_TENSORS_CAP_DEFAULT ";" GST_TENSORS_FLEX_CAP_DEFAULT

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
  PROP_TIMEOUT,
  PROP_OPERATION,
  PROP_MQTT_HOST,
  PROP_MQTT_PORT,
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
          3600, DEFAULT_TIMEOUT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_OPERATION,
      g_param_spec_string ("operation", "Operation",
          "The main operation of the host and option if necessary. "
          "(operation)/(optional topic for main operation).",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_MQTT_HOST,
      g_param_spec_string ("mqtt-host", "MQTT Host",
          "MQTT host address to connect.",
          DEFAULT_MQTT_HOST_ADDRESS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (gobject_class, PROP_MQTT_PORT,
      g_param_spec_string ("mqtt-port", "MQTT Port", "MQTT port to connect.",
          DEFAULT_MQTT_HOST_PORT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  src->operation = NULL;
  src->query_handle = NULL;
  src->mqtt_topic = NULL;
  src->mqtt_host = g_strdup (DEFAULT_MQTT_HOST_ADDRESS);
  src->mqtt_port = g_strdup (DEFAULT_MQTT_HOST_PORT);
  src->mqtt_state = MQTT_INITIALIZING;
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
  src->host = NULL;
  g_free (src->operation);
  src->operation = NULL;
  g_free (src->mqtt_host);
  src->mqtt_host = NULL;
  g_free (src->mqtt_port);
  src->mqtt_port = NULL;
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
      serversrc->timeout = g_value_get_uint (value);
      break;
    case PROP_OPERATION:
      if (!g_value_get_string (value)) {
        nns_logw
            ("operation property cannot be NULL. MQTT-hubrid is disabled.");
        break;
      }
      g_free (serversrc->operation);
      serversrc->operation = g_value_dup_string (value);
      break;
    case PROP_MQTT_HOST:
      if (!g_value_get_string (value)) {
        g_warning ("MQTT host property cannot be NULL");
        break;
      }
      g_free (serversrc->mqtt_host);
      serversrc->mqtt_host = g_value_dup_string (value);
      break;
    case PROP_MQTT_PORT:
      if (!g_value_get_string (value)) {
        g_warning ("MQTT port property cannot be NULL");
        break;
      }
      g_free (serversrc->mqtt_port);
      serversrc->mqtt_host = g_value_dup_string (value);
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
    case PROP_MQTT_HOST:
      g_value_set_string (value, serversrc->mqtt_host);
      break;
    case PROP_MQTT_PORT:
      g_value_set_string (value, serversrc->mqtt_port);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief MQTT State change callback
 */
static void
_state_change_cb (void *user_data, query_mqtt_state_t state)
{
  GstTensorQueryServerSrc *src = (GstTensorQueryServerSrc *) (user_data);
  src->mqtt_state = state;
  nns_logd ("MQTT stated changed to %d", src->mqtt_state);
}

/**
 * @brief start processing of query_serversrc, setting up the server
 */
static gboolean
gst_tensor_query_serversrc_start (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);
  gboolean ret = TRUE;

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
          src->host, src->port, TRUE) != 0) {
    nns_loge ("Failed to setup server");
    return FALSE;
  }
  /** Publish query sever connection info */
  if (src->operation) {
    gint err = 0;
    gchar *device_name = NULL, *msg = NULL;
    gchar *sink_host = NULL;
    guint16 sink_port = 0;

    /**
     * @todo Device name should have unique name. Consider using MAC address later.
     *       Now, use IP and port number temporarily.
    */
    device_name = g_strdup_printf ("device-%s-%u", src->host, src->port);
    src->mqtt_topic =
        g_strdup_printf ("edge/inference/%s/%s/", device_name, src->operation);
    nns_logd ("Query server source mqtt topic: %s", src->mqtt_topic);

    sink_host = gst_tensor_query_server_get_sink_host ();
    if (!sink_host) {
      nns_logw ("sink host is not given. Use default value: localhost");
      sink_host = g_strdup ("localhost");
    }
    sink_port = gst_tensor_query_server_get_sink_port ();
    if (0 == sink_port) {
      nns_logw ("sink port is not given. Use default value: 3000");
      sink_port = 3000;
    }

    msg =
        g_strdup_printf ("%s/%u/%s/%u", src->host, src->port, sink_host,
        sink_port);
    nns_logd ("Query server source publishing msg: %s", msg);

    err =
        query_open_connection (&src->query_handle, src->mqtt_host,
        src->mqtt_port, _state_change_cb, src);
    if (err != 0) {
      nns_loge ("[MQTT] Failed to connect mqtt broker. err: %d\n", err);
      ret = FALSE;
      goto done;
    }

    /** Wait until connection is established. */
    while (MQTT_CONNECTED != src->mqtt_state) {
      g_usleep (10000);
    }

    err =
        query_publish_raw_data (src->query_handle, src->mqtt_topic, msg,
        strlen (msg), TRUE);
    if (err != 0) {
      nns_loge ("[MQTT] Failed to publish raw data. err: %d\n", err);
      ret = FALSE;
      goto done;
    }
  done:
    g_free (msg);
    g_free (device_name);
    g_free (sink_host);
  } else {
    nns_logw ("MQTT-Hybrid is disabled.");
    nns_logw ("Specify operation to register server to mqtt broker.");
    nns_logw ("e.g., operation=object_detection/mobilev3");
  }
  return ret;
}

/**
 * @brief stop processing of query_serversrc
 */
static gboolean
gst_tensor_query_serversrc_stop (GstBaseSrc * bsrc)
{
  GstTensorQueryServerSrc *src = GST_TENSOR_QUERY_SERVERSRC (bsrc);
  nnstreamer_query_server_data_free (src->server_data);
  if (src->query_handle) {
    query_clear_retained_topic (src->query_handle, src->mqtt_topic);
    g_free (src->mqtt_topic);
    src->mqtt_topic = NULL;

    if (0 != query_close_connection (src->query_handle)) {
      nns_loge ("[MQTT] Failed to close connection.");
      return FALSE;
    }
  }

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

    /**
     * Set non-blocking mode to receive the command data.
     * If data is not available in the socket, check the next socket.
     */
    if (0 != nnstreamer_query_receive (conn, &cmd_data, 0)) {
      nns_logi ("Failed to receive cmd");
      continue;
    }

    switch (cmd_data.cmd) {
      case _TENSOR_QUERY_CMD_REQUEST_INFO:
      {
        GstTensorsConfig *config = &cmd_data.data_info.config;
        if ((gst_tensors_config_is_flexible (config) &&
                gst_tensors_config_is_flexible (&src->src_config)) ||
            gst_tensors_info_is_equal (&config->info, &src->src_config.info)) {
          cmd_data.cmd = _TENSOR_QUERY_CMD_RESPOND_APPROVE;
          /* respond sink config */
          gst_tensor_query_server_get_sink_config (config);
        } else {
          /* respond deny with src config */
          nns_logw ("tensor info is not equal");
          cmd_data.cmd = _TENSOR_QUERY_CMD_RESPOND_DENY;
          gst_tensors_config_copy (config, &src->src_config);
        }
        if (nnstreamer_query_send (conn, &cmd_data, src->timeout) != 0) {
          nns_logi ("Failed to send respond");
          continue;
        }
        break;
      }
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

          ecode = nnstreamer_query_receive (conn, &cmd_data, 1);
          gst_memory_unmap (mem, &map);

          if (ecode != 0) {
            nns_logi ("Failed to receive data");
            goto reset_buffer;
          }
        }

        /* receive end */
        if (0 != nnstreamer_query_receive (conn, &cmd_data, 1) ||
            cmd_data.cmd != _TENSOR_QUERY_CMD_TRANSFER_END) {
          nns_logi ("Failed to receive end command");
          goto reset_buffer;
        }

        meta_query = gst_buffer_add_meta_query (*outbuf);
        if (meta_query) {
          meta_query->client_id =
              nnstreamer_query_connection_get_client_id (conn);
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
