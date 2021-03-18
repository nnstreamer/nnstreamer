/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsink.c
 * @date    08 Mar 2021
 * @brief   Publish incoming data streams as a MQTT topic
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef G_OS_WIN32
#include <process.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#include <gst/base/gstbasesink.h>
#include <MQTTClient.h>

#include "mqttsink.h"

static GstStaticPadTemplate sink_pad_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

#define gst_mqtt_sink_parent_class parent_class
G_DEFINE_TYPE (GstMqttSink, gst_mqtt_sink, GST_TYPE_BASE_SINK);

GST_DEBUG_CATEGORY_STATIC (gst_mqtt_sink_debug);
#define GST_CAT_DEFAULT gst_mqtt_sink_debug

enum
{
  PROP_0,

  PROP_MQTT_CLIENT_ID,
  PROP_MQTT_HOST_ADDRESS,
  PROP_MQTT_HOST_PORT,
  PROP_MQTT_PUB_TOPIC,
  PROP_MQTT_PUB_WAIT_TIMEOUT,
  PROP_MQTT_OPT_CLEANSESSION,
  PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL,
  PROP_NUM_BUFFERS,

  PROP_LAST
};

enum
{
  DEFAULT_NUM_BUFFERS = -1,
  DEFAULT_QOS = TRUE,
  DEFAULT_SYNC = FALSE,
  DEFAULT_MQTT_OPT_CLEANSESSION = TRUE,
  DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL = 60,    /* 1 minute */
  DEFAULT_MQTT_DISCONNECT_TIMEOUT = 10000,      /* 10 secs */
  DEFAULT_MQTT_PUB_WAIT_TIMEOUT = 10000,        /* 10 secs */
};

static const gchar DEFAULT_MQTT_HOST_ADDRESS[] = "tcp://localhost";
static const gchar DEFAULT_MQTT_HOST_PORT[] = "1883";
static const gchar TAG_ERR_MQTTSINK[] = "ERROR: MQTTSink";
const gchar *DEFAULT_MQTT_CLIENT_ID;
const gchar *DEFAULT_MQTT_PUB_TOPIC;

/** Function prototype declarations */
static void
gst_mqtt_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void
gst_mqtt_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_mqtt_sink_class_finalize (GObject * object);

static GstStateChangeReturn
gst_mqtt_sink_change_state (GstElement * element, GstStateChange transition);

static gboolean gst_mqtt_sink_start (GstBaseSink * basesink);
static gboolean gst_mqtt_sink_stop (GstBaseSink * basesink);
static gboolean gst_mqtt_sink_query (GstBaseSink * basesink, GstQuery * query);
static GstFlowReturn
gst_mqtt_sink_render (GstBaseSink * basesink, GstBuffer * buffer);
static GstFlowReturn
gst_mqtt_sink_render_list (GstBaseSink * basesink, GstBufferList * list);
static gboolean gst_mqtt_sink_event (GstBaseSink * basesink, GstEvent * event);
static gchar *gst_mqtt_sink_get_client_id (GstMqttSink * self);
static void gst_mqtt_sink_set_client_id (GstMqttSink * self, const gchar * id);
static gchar *gst_mqtt_sink_get_host_address (GstMqttSink * self);
static void gst_mqtt_sink_set_host_address (GstMqttSink * self,
    const gchar * addr);
static gchar *gst_mqtt_sink_get_host_port (GstMqttSink * self);
static void gst_mqtt_sink_set_host_port (GstMqttSink * self,
    const gchar * port);
static gchar *gst_mqtt_sink_get_pub_topic (GstMqttSink * self);
static void gst_mqtt_sink_set_pub_topic (GstMqttSink * self,
    const gchar * topic);
static gulong gst_mqtt_sink_get_pub_wait_timeout (GstMqttSink * self);
static void gst_mqtt_sink_set_pub_wait_timeout (GstMqttSink * self,
    const gulong to);
static gboolean gst_mqtt_sink_get_opt_cleansession (GstMqttSink * self);
static void gst_mqtt_sink_set_opt_cleansession (GstMqttSink * self,
    const gboolean val);
static gint gst_mqtt_sink_get_opt_keep_alive_interval (GstMqttSink * self);
static void gst_mqtt_sink_set_opt_keep_alive_interval (GstMqttSink * self,
    const gint num);

static gint gst_mqtt_sink_get_num_buffers (GstMqttSink * self);
static void gst_mqtt_sink_set_num_buffers (GstMqttSink * self, const gint num);

/**
 * @brief Initialize GstMqttSink object
 */
static void
gst_mqtt_sink_init (GstMqttSink * self)
{
  GstBaseSink *basesink = GST_BASE_SINK (self);
  MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;

  self->gquark_err_tag = g_quark_from_string (TAG_ERR_MQTTSINK);

  self->mqtt_client_handle = g_malloc0 (sizeof (*self->mqtt_client_handle));
  if (!self->mqtt_client_handle) {
    self->err = g_error_new (self->gquark_err_tag, ENOMEM,
        "%s: self->mqtt_client_handle: %s", __func__, g_strerror (ENOMEM));
    return;
  }
  self->mqtt_conn_opts = g_malloc0 (sizeof (*self->mqtt_conn_opts));
  if (!self->mqtt_conn_opts) {
    self->err = g_error_new (self->gquark_err_tag, ENOMEM,
        "%s: self->mqtt_conn_opts: %s", __func__, g_strerror (ENOMEM));
    return;
  }
  self->mqtt_conn_opts = memcpy (self->mqtt_conn_opts, &conn_opts,
      sizeof (conn_opts));

  self->mqtt_msg_hdr = g_malloc0 (sizeof (GST_MQTT_LEN_MSG_HDR));
  if (!self->mqtt_msg_hdr) {
    self->err = g_error_new (self->gquark_err_tag, ENOMEM,
        "%s: self->mqtt_msg_hdr: %s", __func__, g_strerror (ENOMEM));
    return;
  }
  self->mqtt_msg_hdr_update_flag = TRUE;

  /** init mqttsink properties */
  self->num_buffers = DEFAULT_NUM_BUFFERS;
  self->mqtt_client_id = (gchar *) DEFAULT_MQTT_CLIENT_ID;
  self->mqtt_host_address = g_strdup (DEFAULT_MQTT_HOST_ADDRESS);
  self->mqtt_host_port = g_strdup (DEFAULT_MQTT_HOST_PORT);
  self->mqtt_topic = (gchar *) DEFAULT_MQTT_PUB_TOPIC;
  self->mqtt_pub_wait_timeout = DEFAULT_MQTT_PUB_WAIT_TIMEOUT;
  self->mqtt_conn_opts->cleansession = DEFAULT_MQTT_OPT_CLEANSESSION;
  self->mqtt_conn_opts->keepAliveInterval =
      DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL;

  /** init basesink properties */
  gst_base_sink_set_qos_enabled (basesink, DEFAULT_QOS);
  gst_base_sink_set_sync (basesink, DEFAULT_SYNC);
}

/**
 * @brief Initialize GstMqttSinkClass object
 */
static void
gst_mqtt_sink_class_init (GstMqttSinkClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSinkClass *gstbasesink_class = GST_BASE_SINK_CLASS (klass);

  DEFAULT_MQTT_CLIENT_ID = g_strdup_printf ("%s/%u/%u", g_get_host_name (),
      getpid (), g_random_int_range (0, 0xFF));
  DEFAULT_MQTT_PUB_TOPIC = g_strdup_printf ("%s/topic", DEFAULT_MQTT_CLIENT_ID);

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, GST_MQTT_ELEM_NAME_SINK, 0,
      "MQTT sink");

  gobject_class->set_property = gst_mqtt_sink_set_property;
  gobject_class->get_property = gst_mqtt_sink_get_property;
  gobject_class->finalize = gst_mqtt_sink_class_finalize;

  g_object_class_install_property (gobject_class, PROP_MQTT_CLIENT_ID,
      g_param_spec_string ("client-id", "Client ID",
          "The client identifier passed to the server (broker)",
          DEFAULT_MQTT_CLIENT_ID, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_HOST_ADDRESS,
      g_param_spec_string ("host", "Host", "Host (broker) to connect to",
          DEFAULT_MQTT_HOST_ADDRESS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_HOST_PORT,
      g_param_spec_string ("port", "Port",
          "Network port of host (broker) to connect to", DEFAULT_MQTT_HOST_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_PUB_TOPIC,
      g_param_spec_string ("pub-topic", "Topic to Publish",
          "The topic's name to publish", DEFAULT_MQTT_PUB_TOPIC,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class,
      PROP_MQTT_PUB_WAIT_TIMEOUT,
      g_param_spec_ulong ("pub-wait-timeout", "Timeout for SyncPublish",
          "Timeout for synchronize execution of the main thread with completed publication of a message",
          1UL, G_MAXULONG, DEFAULT_MQTT_PUB_WAIT_TIMEOUT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_OPT_CLEANSESSION,
      g_param_spec_boolean ("cleansession", "Cleansession",
          "When it is TRUE, the state information is discarded at connect and disconnect.",
          DEFAULT_MQTT_OPT_CLEANSESSION,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class,
      PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL,
      g_param_spec_int ("keep-alive-interval", "Keep Alive Interval",
          "The maximum time (in seconds) that should pass without communication between the client and the server (broker)",
          1, G_MAXINT32, DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_BUFFERS,
      g_param_spec_int ("num-buffers", "Num Buffers",
          "Number of (remaining) buffers to accept until sending EOS event (-1 = no limit)",
          -1, G_MAXINT32, DEFAULT_NUM_BUFFERS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gstelement_class->change_state = gst_mqtt_sink_change_state;

  gstbasesink_class->start = GST_DEBUG_FUNCPTR (gst_mqtt_sink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR (gst_mqtt_sink_stop);
  gstbasesink_class->query = GST_DEBUG_FUNCPTR (gst_mqtt_sink_query);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR (gst_mqtt_sink_render);
  gstbasesink_class->render_list =
      GST_DEBUG_FUNCPTR (gst_mqtt_sink_render_list);
  gstbasesink_class->event = GST_DEBUG_FUNCPTR (gst_mqtt_sink_event);

  gst_element_class_set_static_metadata (gstelement_class,
      "MQTT sink", "Sink/MQTT",
      "Publish incoming data streams as a MQTT topic",
      "Wook Song <wook16.song@samsung.com>");
  gst_element_class_add_static_pad_template (gstelement_class,
      &sink_pad_template);
}

/**
 * @brief The setter for the mqttsink's properties
 */
static void
gst_mqtt_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMqttSink *self = GST_MQTT_SINK (object);

  switch (prop_id) {
    case PROP_MQTT_CLIENT_ID:
      gst_mqtt_sink_set_client_id (self, g_value_get_string (value));
      break;
    case PROP_MQTT_HOST_ADDRESS:
      gst_mqtt_sink_set_host_address (self, g_value_get_string (value));
      break;
    case PROP_MQTT_HOST_PORT:
      gst_mqtt_sink_set_host_port (self, g_value_get_string (value));
      break;
    case PROP_MQTT_PUB_TOPIC:
      gst_mqtt_sink_set_pub_topic (self, g_value_get_string (value));
      break;
    case PROP_MQTT_PUB_WAIT_TIMEOUT:
      gst_mqtt_sink_set_pub_wait_timeout (self, g_value_get_ulong (value));
      break;
    case PROP_MQTT_OPT_CLEANSESSION:
      gst_mqtt_sink_set_opt_cleansession (self, g_value_get_boolean (value));
      break;
    case PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL:
      gst_mqtt_sink_set_opt_keep_alive_interval (self, g_value_get_int (value));
      break;
    case PROP_NUM_BUFFERS:
      gst_mqtt_sink_set_num_buffers (self, g_value_get_int (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief The getter for the mqttsink's properties
 */
static void
gst_mqtt_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstMqttSink *self = GST_MQTT_SINK (object);

  switch (prop_id) {
    case PROP_MQTT_CLIENT_ID:
      g_value_set_string (value, gst_mqtt_sink_get_client_id (self));
      break;
    case PROP_MQTT_HOST_ADDRESS:
      g_value_set_string (value, gst_mqtt_sink_get_host_address (self));
      break;
    case PROP_MQTT_HOST_PORT:
      g_value_set_string (value, gst_mqtt_sink_get_host_port (self));
      break;
    case PROP_MQTT_PUB_TOPIC:
      g_value_set_string (value, gst_mqtt_sink_get_pub_topic (self));
      break;
    case PROP_MQTT_PUB_WAIT_TIMEOUT:
      g_value_set_ulong (value, gst_mqtt_sink_get_pub_wait_timeout (self));
      break;
    case PROP_MQTT_OPT_CLEANSESSION:
      g_value_set_boolean (value, gst_mqtt_sink_get_opt_cleansession (self));
      break;
    case PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL:
      g_value_set_int (value, gst_mqtt_sink_get_opt_keep_alive_interval (self));
      break;
    case PROP_NUM_BUFFERS:
      g_value_set_int (value, gst_mqtt_sink_get_num_buffers (self));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Finalize GstMqttSinkClass object
 */
static void
gst_mqtt_sink_class_finalize (GObject * object)
{
  GstMqttSink *self = GST_MQTT_SINK (object);

  g_free (self->mqtt_host_address);
  g_free (self->mqtt_host_port);
  g_free (self->mqtt_client_handle);
  g_free (self->mqtt_conn_opts);
  g_free (self->mqtt_msg_hdr);
  if (self->err)
    g_error_free (self->err);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Handle mqttsink's state change
 */
static GstStateChangeReturn
gst_mqtt_sink_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstMqttSink *self = GST_MQTT_SINK (element);

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_NULL_TO_READY");
      if (self->err) {
        g_printerr ("%s: %s\n", g_quark_to_string (self->err->domain),
            self->err->message);
        return GST_STATE_CHANGE_FAILURE;
      }
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_READY_TO_PAUSED");
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_PAUSED_TO_PLAYING");
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_PLAYING_TO_PAUSED");
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_PAUSED_TO_READY");
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_READY_TO_NULL");
    default:
      break;
  }

  return ret;
}

/**
 * @brief Start mqttsink, called when state changed null to ready
 */
static gboolean
gst_mqtt_sink_start (GstBaseSink * basesink)
{
  GstMqttSink *self = GST_MQTT_SINK (basesink);
  gchar *haddr = g_strdup_printf ("%s:%s", self->mqtt_host_address,
      self->mqtt_host_port);
  int ret;

  /**
   * @todo Support other persistence mechanisms
   *    MQTTCLIENT_PERSISTENCE_NONE: A memory-based persistence mechanism
   *    MQTTCLIENT_PERSISTENCE_DEFAULT: The default file system-based
   *                                    persistence mechanism
   *    MQTTCLIENT_PERSISTENCE_USER: An application-specific persistence
   *                                 mechanism
   */
  ret = MQTTClient_create (self->mqtt_client_handle, haddr,
      self->mqtt_client_id, MQTTCLIENT_PERSISTENCE_DEFAULT, NULL);
  g_free (haddr);
  if (ret != MQTTCLIENT_SUCCESS)
    return FALSE;

  ret = MQTTClient_connect (*self->mqtt_client_handle, self->mqtt_conn_opts);
  if (ret != MQTTCLIENT_SUCCESS) {
    MQTTClient_destroy (self->mqtt_client_handle);
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief Stop mqttsink, called when state changed ready to null
 */
static gboolean
gst_mqtt_sink_stop (GstBaseSink * basesink)
{
  GstMqttSink *self = GST_MQTT_SINK (basesink);

  MQTTClient_disconnect (*self->mqtt_client_handle,
      DEFAULT_MQTT_DISCONNECT_TIMEOUT);
  MQTTClient_destroy (self->mqtt_client_handle);

  return TRUE;
}

/**
 * @brief Perform queries on the element
 */
static gboolean
gst_mqtt_sink_query (GstBaseSink * basesink, GstQuery * query)
{
  gboolean ret = FALSE;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_SEEKING:{
      GstFormat fmt;

      /* GST_QUERY_SEEKING is not supported */
      gst_query_parse_seeking (query, &fmt, NULL, NULL, NULL);
      gst_query_set_seeking (query, fmt, FALSE, 0, -1);
      ret = TRUE;
      break;
    }
    default:{
      ret = GST_BASE_SINK_CLASS (parent_class)->query (basesink, query);
      break;
    }
  }

  return ret;
}

/**
 * @brief The callback to process each buffer receiving on the sink pad
 */
static GstFlowReturn
gst_mqtt_sink_render (GstBaseSink * basesink, GstBuffer * in_buf)
{
  MQTTClient_message pubmsg = MQTTClient_message_initializer;
  MQTTClient_deliveryToken token;
  GstMqttSink *self = GST_MQTT_SINK (basesink);
  GstMQTTMessageHdr *mqtt_msg_hdr;
  GstBuffer *pubmsg_buf;
  GstMemory *hdr_mem;
  GstMemory *payload_mem;
  GstMapInfo payload_mem_map;
  GstFlowReturn ret = GST_FLOW_OK;
  gsize payload_len;
  gint mqtt_rc;
  guint i;

  GST_OBJECT_LOCK (self);
  if (self->num_buffers == 0)
    goto ret_eos;

  if (self->num_buffers != -1)
    self->num_buffers -= 1;
  GST_OBJECT_UNLOCK (self);

  /** Create a new empty GstBuffer since in_buf is not writeable */
  pubmsg_buf = gst_buffer_new ();
  if (!pubmsg_buf) {
    ret = GST_FLOW_ERROR;
    goto ret_err;
  }

  payload_len = 0;
  mqtt_msg_hdr = self->mqtt_msg_hdr;
  GST_OBJECT_LOCK (self);
  mqtt_msg_hdr->num_mems = gst_buffer_n_memory (in_buf);
  GST_OBJECT_UNLOCK (self);
  for (i = 0; i < mqtt_msg_hdr->num_mems; ++i) {
    GstMemory *each_mem;

    each_mem = gst_buffer_peek_memory (in_buf, i);
    if (!each_mem) {
      ret = GST_FLOW_ERROR;
      goto ret_err_unref_pub_buf;
    }
    GST_OBJECT_LOCK (self);
    mqtt_msg_hdr->size_mems[i] = each_mem->size;
    GST_OBJECT_UNLOCK (self);
    payload_len += each_mem->size;
    gst_memory_ref (each_mem);
    gst_buffer_append_memory (pubmsg_buf, each_mem);
  }

  hdr_mem = gst_memory_new_wrapped (0, mqtt_msg_hdr, GST_MQTT_LEN_MSG_HDR, 0,
      GST_MQTT_LEN_MSG_HDR, NULL, NULL);
  if (!hdr_mem) {
    ret = GST_FLOW_ERROR;
    goto ret_err_unref_pub_buf;
  }

  payload_len += GST_MQTT_LEN_MSG_HDR;
  gst_buffer_prepend_memory (pubmsg_buf, hdr_mem);

  payload_mem = gst_buffer_get_all_memory (pubmsg_buf);
  if (!gst_memory_map (payload_mem, &payload_mem_map, GST_MAP_READ)) {
    ret = GST_FLOW_ERROR;
    goto ret_err_unref_pub_buf;
  }

  pubmsg.payload = payload_mem_map.data;
  /** the data type of payloadlen is int */
  pubmsg.payloadlen = (gint) payload_len;
  /** @todo MQTTClient_message's properties should be adjustable. */
  pubmsg.qos = 1;
  pubmsg.retained = 0;

  mqtt_rc = MQTTClient_publishMessage (*self->mqtt_client_handle,
      self->mqtt_topic, &pubmsg, &token);
  if (mqtt_rc != MQTTCLIENT_SUCCESS) {
    ret = GST_FLOW_ERROR;
    goto ret_err_cleanup_payload;
  }

  MQTTClient_waitForCompletion (*self->mqtt_client_handle, token,
      self->mqtt_pub_wait_timeout);

  GST_DEBUG_OBJECT (self, "Message with delivery token %d delivered\n", token);

ret_err_cleanup_payload:
  gst_memory_unmap (payload_mem, &payload_mem_map);
  gst_memory_unref (payload_mem);

ret_err_unref_pub_buf:
  gst_buffer_unref (pubmsg_buf);

ret_err:
  return ret;

ret_eos:
  GST_OBJECT_UNLOCK (self);
  return GST_FLOW_EOS;
}

/**
 * @brief The callback to process GstBufferList (instead of a single buffer)
 *        on the sink pad
 */
static GstFlowReturn
gst_mqtt_sink_render_list (GstBaseSink * basesink, GstBufferList * list)
{
  guint num_buffers = gst_buffer_list_length (list);
  GstFlowReturn ret;
  GstBuffer *buffer;
  guint i;

  for (i = 0; i < num_buffers; ++i) {
    buffer = gst_buffer_list_get (list, i);
    ret = gst_mqtt_sink_render (basesink, buffer);
    if (ret != GST_FLOW_OK)
      break;
  }

  return ret;
}

/**
 * @brief Handle events arriving on the sink pad
 */
static gboolean
gst_mqtt_sink_event (GstBaseSink * basesink, GstEvent * event)
{
  GstEventType type = GST_EVENT_TYPE (event);
  gboolean ret = FALSE;

  switch (type) {
    default:
      ret = GST_BASE_SINK_CLASS (parent_class)->event (basesink, event);
      break;
  }

  return ret;
}

/**
 * @brief Getter for the 'client-id' property.
 */
static gchar *
gst_mqtt_sink_get_client_id (GstMqttSink * self)
{
  return self->mqtt_client_id;
}

/**
 * @brief Setter for the 'client-id' property.
 */
static void
gst_mqtt_sink_set_client_id (GstMqttSink * self, const gchar * id)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_client_id = g_strdup (id);
  GST_OBJECT_UNLOCK (self);
  g_free ((void *) DEFAULT_MQTT_CLIENT_ID);
}

/**
 * @brief Getter for the 'host' property.
 */
static gchar *
gst_mqtt_sink_get_host_address (GstMqttSink * self)
{
  return self->mqtt_host_address;
}

/**
 * @brief Setter for the 'host' property
 */
static void
gst_mqtt_sink_set_host_address (GstMqttSink * self, const gchar * addr)
{
  /**
   * @todo Handle the case where the addr is changed at runtime
   */
  GST_OBJECT_LOCK (self);
  self->mqtt_host_address = g_strdup (addr);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'port' property.
 */
static gchar *
gst_mqtt_sink_get_host_port (GstMqttSink * self)
{
  return self->mqtt_host_port;
}

/**
 * @brief Setter for the 'port' property
 */
static void
gst_mqtt_sink_set_host_port (GstMqttSink * self, const gchar * port)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_host_port = g_strdup (port);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'pub-topic' property
 */
static gchar *
gst_mqtt_sink_get_pub_topic (GstMqttSink * self)
{
  return self->mqtt_topic;
}

/**
 * @brief Setter for the 'pub-topic' property
 */
static void
gst_mqtt_sink_set_pub_topic (GstMqttSink * self, const gchar * topic)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_topic = g_strdup (topic);
  GST_OBJECT_UNLOCK (self);
  g_free ((void *) DEFAULT_MQTT_PUB_TOPIC);
}

/**
 * @brief Getter for the 'cleansession' property.
 */
static gboolean
gst_mqtt_sink_get_opt_cleansession (GstMqttSink * self)
{
  return self->mqtt_conn_opts->cleansession;
}

/**
 * @brief Setter for the 'cleansession' property.
 */
static void
gst_mqtt_sink_set_opt_cleansession (GstMqttSink * self, const gboolean val)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_conn_opts->cleansession = val;
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'pub-wait-timeout' property.
 */
static gulong
gst_mqtt_sink_get_pub_wait_timeout (GstMqttSink * self)
{
  return self->mqtt_pub_wait_timeout;
}

/**
 * @brief Setter for the 'pub-wait-timeout' property.
 */
static void
gst_mqtt_sink_set_pub_wait_timeout (GstMqttSink * self, const gulong to)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_pub_wait_timeout = to;
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'keep-alive-interval' property
 */
static gint
gst_mqtt_sink_get_opt_keep_alive_interval (GstMqttSink * self)
{
  return self->mqtt_conn_opts->keepAliveInterval;
}

/**
 * @brief Setter for the 'keep-alive-interval' property
 */
static void
gst_mqtt_sink_set_opt_keep_alive_interval (GstMqttSink * self, const gint num)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_conn_opts->keepAliveInterval = num;
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'num-buffers' property.
 */
static gint
gst_mqtt_sink_get_num_buffers (GstMqttSink * self)
{
  gint num_buffers;

  GST_OBJECT_LOCK (self);
  num_buffers = self->num_buffers;
  GST_OBJECT_UNLOCK (self);

  return num_buffers;
}

/**
 * @brief Setter for the 'num-buffers' property
 */
static void
gst_mqtt_sink_set_num_buffers (GstMqttSink * self, const gint num)
{
  GST_OBJECT_LOCK (self);
  self->num_buffers = num;
  GST_OBJECT_UNLOCK (self);
}
