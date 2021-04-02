/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsrc.c
 * @date    08 Mar 2021
 * @brief   Subscribe a MQTT topic and push incoming data to the GStreamer pipeline
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

#include <gst/base/gstbasesrc.h>
#include <MQTTAsync.h>

#include "mqttsrc.h"

static GstStaticPadTemplate src_pad_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

#define gst_mqtt_src_parent_class parent_class
G_DEFINE_TYPE (GstMqttSrc, gst_mqtt_src, GST_TYPE_BASE_SRC);

GST_DEBUG_CATEGORY_STATIC (gst_mqtt_src_debug);
#define GST_CAT_DEFAULT gst_mqtt_src_debug

enum
{
  PROP_0,

  PROP_MQTT_CLIENT_ID,
  PROP_MQTT_HOST_ADDRESS,
  PROP_MQTT_HOST_PORT,
  PROP_MQTT_SUB_TOPIC,
  PROP_MQTT_SUB_TIMEOUT,
  PROP_MQTT_OPT_CLEANSESSION,
  PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL,

  PROP_LAST
};

enum
{
  DEFAULT_MQTT_OPT_CLEANSESSION = TRUE,
  DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL = 1000000,       /* 1 minute */
  DEFAULT_MQTT_SUB_TIMEOUT = 10000000,  /* 10 seconds */
  DEFAULT_MQTT_SUB_TIMEOUT_MIN = 1000000,       /* 1 seconds */
};

static const gchar DEFAULT_MQTT_HOST_ADDRESS[] = "tcp://localhost";
static const gchar DEFAULT_MQTT_HOST_PORT[] = "1883";
static const gchar TAG_ERR_MQTTSRC[] = "ERROR: MQTTSrc";
const gchar *DEFAULT_MQTT_CLIENT_ID;
const gchar *DEFAULT_MQTT_SUB_TOPIC;

/** Function prototype declarations */
static void
gst_mqtt_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void
gst_mqtt_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_mqtt_src_class_finalize (GObject * object);

static GstStateChangeReturn
gst_mqtt_src_change_state (GstElement * element, GstStateChange transition);

static gboolean gst_mqtt_src_start (GstBaseSrc * basesrc);
static gboolean gst_mqtt_src_stop (GstBaseSrc * basesrc);
static GstCaps *gst_mqtt_src_get_caps (GstBaseSrc * basesrc, GstCaps * filter);

static void
gst_mqtt_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end);
static gboolean gst_mqtt_src_is_seekable (GstBaseSrc * basesrc);
static GstFlowReturn
gst_mqtt_src_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf);

static gchar *gst_mqtt_src_get_client_id (GstMqttSrc * self);
static void gst_mqtt_src_set_client_id (GstMqttSrc * self, const gchar * id);
static gchar *gst_mqtt_src_get_host_address (GstMqttSrc * self);
static void gst_mqtt_src_set_host_address (GstMqttSrc * self,
    const gchar * addr);
static gchar *gst_mqtt_src_get_host_port (GstMqttSrc * self);
static void gst_mqtt_src_set_host_port (GstMqttSrc * self, const gchar * port);
static gint64 gst_mqtt_src_get_sub_timeout (GstMqttSrc * self);
static void gst_mqtt_src_set_sub_timeout (GstMqttSrc * self, const gint64 t);
static gchar *gst_mqtt_src_get_sub_topic (GstMqttSrc * self);
static void gst_mqtt_src_set_sub_topic (GstMqttSrc * self, const gchar * topic);
static gboolean gst_mqtt_src_get_opt_cleansession (GstMqttSrc * self);
static void gst_mqtt_src_set_opt_cleansession (GstMqttSrc * self,
    const gboolean val);
static gint gst_mqtt_src_get_opt_keep_alive_interval (GstMqttSrc * self);
static void gst_mqtt_src_set_opt_keep_alive_interval (GstMqttSrc * self,
    const gint num);

static void cb_mqtt_on_connection_lost (void *context, char *cause);
static void cb_mqtt_on_delivery_complete (void *context, MQTTAsync_token token);
static int cb_mqtt_on_message_arrived (void *context, char *topic_name,
    int topic_len, MQTTAsync_message * message);
static void cb_mqtt_on_connect (void *context,
    MQTTAsync_successData * response);
static void cb_mqtt_on_connect_failure (void *context,
    MQTTAsync_failureData * response);
static void cb_mqtt_on_subscribe (void *context,
    MQTTAsync_successData * response);
static void cb_mqtt_on_subscribe_failure (void *context,
    MQTTAsync_failureData * response);

static void cb_memory_wrapped_destroy (void *p);

static GstMQTTMessageHdr *_extract_mqtt_msg_hdr_from (GstMemory * mem,
    GstMemory ** hdr_mem, GstMapInfo * hdr_map_info);


/** Function defintions */
/**
 * @brief Initialize GstMqttSrc object
 */
static void
gst_mqtt_src_init (GstMqttSrc * self)
{
  MQTTAsync_connectOptions conn_opts = MQTTAsync_connectOptions_initializer;

  self->gquark_err_tag = g_quark_from_string (TAG_ERR_MQTTSRC);

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
  /** init mqttsrc properties */
  self->mqtt_client_id = (gchar *) DEFAULT_MQTT_CLIENT_ID;
  self->mqtt_host_address = g_strdup (DEFAULT_MQTT_HOST_ADDRESS);
  self->mqtt_host_port = g_strdup (DEFAULT_MQTT_HOST_PORT);
  self->mqtt_topic = (gchar *) DEFAULT_MQTT_SUB_TOPIC;
  self->mqtt_sub_timeout = (gint64) DEFAULT_MQTT_SUB_TIMEOUT;
  self->mqtt_conn_opts->cleansession = DEFAULT_MQTT_OPT_CLEANSESSION;
  self->mqtt_conn_opts->keepAliveInterval =
      DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL;
  self->mqtt_conn_opts->onSuccess = cb_mqtt_on_connect;
  self->mqtt_conn_opts->onFailure = cb_mqtt_on_connect_failure;

  /** init private member variables */
  self->aqueue = g_async_queue_new ();
  self->is_connected = FALSE;
  self->is_subscribed = FALSE;
  g_cond_init (&self->gcond);
}

/**
 * @brief Initialize GstMqttSrcClass object
 */
static void
gst_mqtt_src_class_init (GstMqttSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);

  DEFAULT_MQTT_CLIENT_ID = g_strdup_printf ("%s/%u/%u", g_get_host_name (),
      getpid (), g_random_int_range (0, 0xFF));
  DEFAULT_MQTT_SUB_TOPIC = g_strdup_printf ("%s/topic", DEFAULT_MQTT_CLIENT_ID);

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, GST_MQTT_ELEM_NAME_SRC, 0,
      "MQTT src");

  gobject_class->set_property = gst_mqtt_src_set_property;
  gobject_class->get_property = gst_mqtt_src_get_property;
  gobject_class->finalize = gst_mqtt_src_class_finalize;

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

  g_object_class_install_property (gobject_class, PROP_MQTT_SUB_TIMEOUT,
      g_param_spec_int64 ("sub-timeout", "Timeout for receiving a message",
          "The timeout (in microseconds) for receiving a message from subscribed topic",
          1000000, G_MAXINT64, DEFAULT_MQTT_SUB_TIMEOUT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_SUB_TOPIC,
      g_param_spec_string ("sub-topic", "Topic to Subscribe",
          "The topic's name to subscribe", DEFAULT_MQTT_SUB_TOPIC,
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

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_mqtt_src_change_state);

  gstbasesrc_class->start = GST_DEBUG_FUNCPTR (gst_mqtt_src_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_mqtt_src_stop);
  gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_caps);
  gstbasesrc_class->get_times = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_times);
  gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR (gst_mqtt_src_is_seekable);
  gstbasesrc_class->create = GST_DEBUG_FUNCPTR (gst_mqtt_src_create);

  gst_element_class_set_static_metadata (gstelement_class,
      "MQTT Source", "Source/MQTT",
      "Subscribe a MQTT topic and push incoming data to the GStreamer pipeline",
      "Wook Song <wook16.song@samsung.com>");
  gst_element_class_add_static_pad_template (gstelement_class,
      &src_pad_template);
}

/**
 * @brief The setter for the mqttsrc's properties
 */
static void
gst_mqtt_src_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstMqttSrc *self = GST_MQTT_SRC (object);

  switch (prop_id) {
    case PROP_MQTT_CLIENT_ID:
      gst_mqtt_src_set_client_id (self, g_value_get_string (value));
      break;
    case PROP_MQTT_HOST_ADDRESS:
      gst_mqtt_src_set_host_address (self, g_value_get_string (value));
      break;
    case PROP_MQTT_HOST_PORT:
      gst_mqtt_src_set_host_port (self, g_value_get_string (value));
      break;
    case PROP_MQTT_SUB_TIMEOUT:
      gst_mqtt_src_set_sub_timeout (self, g_value_get_int64 (value));
      break;
    case PROP_MQTT_SUB_TOPIC:
      gst_mqtt_src_set_sub_topic (self, g_value_get_string (value));
      break;
    case PROP_MQTT_OPT_CLEANSESSION:
      gst_mqtt_src_set_opt_cleansession (self, g_value_get_boolean (value));
      break;
    case PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL:
      gst_mqtt_src_set_opt_keep_alive_interval (self, g_value_get_int (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief The getter for the mqttsrc's properties
 */
static void
gst_mqtt_src_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstMqttSrc *self = GST_MQTT_SRC (object);

  switch (prop_id) {
    case PROP_MQTT_CLIENT_ID:
      g_value_set_string (value, gst_mqtt_src_get_client_id (self));
      break;
    case PROP_MQTT_HOST_ADDRESS:
      g_value_set_string (value, gst_mqtt_src_get_host_address (self));
      break;
    case PROP_MQTT_HOST_PORT:
      g_value_set_string (value, gst_mqtt_src_get_host_port (self));
      break;
    case PROP_MQTT_SUB_TIMEOUT:
      g_value_set_int64 (value, gst_mqtt_src_get_sub_timeout (self));
      break;
    case PROP_MQTT_SUB_TOPIC:
      g_value_set_string (value, gst_mqtt_src_get_sub_topic (self));
      break;
    case PROP_MQTT_OPT_CLEANSESSION:
      g_value_set_boolean (value, gst_mqtt_src_get_opt_cleansession (self));
      break;
    case PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL:
      g_value_set_int (value, gst_mqtt_src_get_opt_keep_alive_interval (self));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Finalize GstMqttSrcClass object
 */
static void
gst_mqtt_src_class_finalize (GObject * object)
{
  GstMqttSrc *self = GST_MQTT_SRC (object);

  g_free (self->mqtt_host_address);
  g_free (self->mqtt_host_port);
  g_free (self->mqtt_client_handle);
  g_free (self->mqtt_conn_opts);
  if (self->err)
    g_error_free (self->err);

  g_clear_pointer (&self->aqueue, g_async_queue_unref);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Handle mqttsrc's state change
 */
static GstStateChangeReturn
gst_mqtt_src_change_state (GstElement * element, GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstMqttSrc *self = GST_MQTT_SRC (element);

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
 * @brief Start mqttsrc, called when state changed null to ready
 */
static gboolean
gst_mqtt_src_start (GstBaseSrc * basesrc)
{
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);
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
  ret = MQTTAsync_create (self->mqtt_client_handle, haddr,
      self->mqtt_client_id, MQTTCLIENT_PERSISTENCE_DEFAULT, NULL);
  g_free (haddr);
  if (ret != MQTTASYNC_SUCCESS)
    return FALSE;

  MQTTAsync_setCallbacks (*self->mqtt_client_handle, self,
      cb_mqtt_on_connection_lost, cb_mqtt_on_message_arrived,
      cb_mqtt_on_delivery_complete);

  self->mqtt_conn_opts->context = self;
  ret = MQTTAsync_connect (*self->mqtt_client_handle, self->mqtt_conn_opts);
  if (ret != MQTTASYNC_SUCCESS)
    return FALSE;
  return TRUE;
}

/**
 * @brief Stop mqttsrc, called when state changed ready to null
 */
static gboolean
gst_mqtt_src_stop (GstBaseSrc * basesrc)
{
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);

  /* todo */
  MQTTAsync_disconnect (*self->mqtt_client_handle, NULL);
  MQTTAsync_destroy (self->mqtt_client_handle);

  return TRUE;
}

/**
 * @brief Get caps of subclass
 */
static GstCaps *
gst_mqtt_src_get_caps (GstBaseSrc * basesrc, GstCaps * filter)
{
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);
  GstPad *pad = basesrc->srcpad;
  GstCaps *cur_caps = gst_pad_get_current_caps (pad);
  GstCaps *caps = gst_caps_new_any ();

  GST_OBJECT_LOCK (self);
  if (cur_caps) {
    GstCaps *intersection =
        gst_caps_intersect_full (cur_caps, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (cur_caps);
    gst_caps_unref (caps);
    caps = intersection;
  }
  GST_OBJECT_UNLOCK (self);

  return caps;
}

/**
 * @brief Return the time information of the given buffer
 */
static void
gst_mqtt_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
  return;
}

/**
 * @brief Check if source supports seeking
 * @note Seeking is not supported since this element handles live subscription data.
 */
static gboolean
gst_mqtt_src_is_seekable (GstBaseSrc * basesrc)
{
  return FALSE;
}

/**
 * @brief Create a buffer containing the subscribed data
 */
static GstFlowReturn
gst_mqtt_src_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf)
{
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);
  gint64 elapsed = self->mqtt_sub_timeout;

  GST_OBJECT_LOCK (self);
  while ((!self->is_connected) || (!self->is_subscribed)) {
    g_cond_wait (&self->gcond, GST_OBJECT_GET_LOCK (self));
    if (self->err) {
      GST_OBJECT_UNLOCK (self);
      goto ret_flow_err;
    }
  }
  GST_OBJECT_UNLOCK (self);

  while (elapsed > 0) {
    *buf = g_async_queue_timeout_pop (self->aqueue,
        DEFAULT_MQTT_SUB_TIMEOUT_MIN);
    if (*buf || self->err)
      break;
    elapsed = elapsed - DEFAULT_MQTT_SUB_TIMEOUT_MIN;
  }

  if (*buf == NULL) {
    /** @todo: Send EoS here */
    if (!self->err)
      self->err = g_error_new (self->gquark_err_tag, GST_FLOW_EOS,
          "%s: Timeout for receiving a message has been expired. Regarding as an error",
          __func__);
    goto ret_flow_err;
  }

  return GST_FLOW_OK;

ret_flow_err:
  if (self->err) {
    g_printerr ("%s: %s\n", g_quark_to_string (self->err->domain),
        self->err->message);
  }
  return GST_FLOW_ERROR;
}

/**
 * @brief Getter for the 'client-id' property.
 */
static gchar *
gst_mqtt_src_get_client_id (GstMqttSrc * self)
{
  return self->mqtt_client_id;
}

/**
 * @brief Setter for the 'client-id' property.
 */
static void
gst_mqtt_src_set_client_id (GstMqttSrc * self, const gchar * id)
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
gst_mqtt_src_get_host_address (GstMqttSrc * self)
{
  return self->mqtt_host_address;
}

/**
 * @brief Setter for the 'host' property
 */
static void
gst_mqtt_src_set_host_address (GstMqttSrc * self, const gchar * addr)
{
  /**
   * @todo Handle the case where the addr is changed at runtime
   */
  g_free (self->mqtt_host_address);
  GST_OBJECT_LOCK (self);
  self->mqtt_host_address = g_strdup (addr);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'port' property.
 */
static gchar *
gst_mqtt_src_get_host_port (GstMqttSrc * self)
{
  return self->mqtt_host_port;
}

/**
 * @brief Setter for the 'port' property
 */
static void
gst_mqtt_src_set_host_port (GstMqttSrc * self, const gchar * port)
{
  g_free (self->mqtt_host_port);
  GST_OBJECT_LOCK (self);
  self->mqtt_host_port = g_strdup (port);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'sub-timeout' property
 */
static gint64
gst_mqtt_src_get_sub_timeout (GstMqttSrc * self)
{
  return self->mqtt_sub_timeout;
}

/**
 * @brief Setter for the 'sub-timeout' property
 */
static void
gst_mqtt_src_set_sub_timeout (GstMqttSrc * self, const gint64 t)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_sub_timeout = t;
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'sub-topic' property
 */
static gchar *
gst_mqtt_src_get_sub_topic (GstMqttSrc * self)
{
  return self->mqtt_topic;
}

/**
 * @brief Setter for the 'sub-topic' property
 */
static void
gst_mqtt_src_set_sub_topic (GstMqttSrc * self, const gchar * topic)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_topic = g_strdup (topic);
  GST_OBJECT_UNLOCK (self);
  g_free ((void *) DEFAULT_MQTT_SUB_TOPIC);
}

/**
 * @brief Getter for the 'cleansession' property.
 */
static gboolean
gst_mqtt_src_get_opt_cleansession (GstMqttSrc * self)
{
  return self->mqtt_conn_opts->cleansession;
}

/**
 * @brief Setter for the 'cleansession' property.
 */
static void
gst_mqtt_src_set_opt_cleansession (GstMqttSrc * self, const gboolean val)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_conn_opts->cleansession = val;
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for the 'keep-alive-interval' property
 */
static gint
gst_mqtt_src_get_opt_keep_alive_interval (GstMqttSrc * self)
{
  return self->mqtt_conn_opts->keepAliveInterval;
}

/**
 * @brief Setter for the 'keep-alive-interval' property
 */
static void
gst_mqtt_src_set_opt_keep_alive_interval (GstMqttSrc * self, const gint num)
{
  GST_OBJECT_LOCK (self);
  self->mqtt_conn_opts->keepAliveInterval = num;
  GST_OBJECT_UNLOCK (self);
}

/**
  * @brief A callback to handle the connection lost to the broker
  */
static void
cb_mqtt_on_connection_lost (void *context, char *cause)
{
  GstMqttSrc *self = GST_MQTT_SRC_CAST (context);

  GST_OBJECT_LOCK (self);
  self->is_connected = FALSE;
  self->is_subscribed = FALSE;
  g_cond_broadcast (&self->gcond);
  self->err = g_error_new (self->gquark_err_tag, EHOSTDOWN,
      "Connection to the host (broker) has been lost: %s",
      g_strerror (EHOSTDOWN));
  GST_OBJECT_UNLOCK (self);
}

/**
  * @brief A callback to handle the post-processing of the delivered message
  * @todo Fill the function body
  */
static void
cb_mqtt_on_delivery_complete (void *context, MQTTAsync_token token)
{

}

/**
  * @brief A callback to handle the arrived message
  */
static int
cb_mqtt_on_message_arrived (void *context, char *topic_name, int topic_len,
    MQTTAsync_message * message)
{
  const int size = message->payloadlen;
  guint8 *data = message->payload;
  GstMQTTMessageHdr *mqtt_msg_hdr;
  GstMapInfo hdr_map_info;
  GstMemory *recieved_mem;
  GstMemory *hdr_mem;
  GstBuffer *buffer;
  GstMqttSrc *self;
  gsize offset;
  guint i;

  self = GST_MQTT_SRC_CAST (context);
  recieved_mem = gst_memory_new_wrapped (0, data, size, 0, size, message,
      (GDestroyNotify) cb_memory_wrapped_destroy);
  if (!recieved_mem) {
    self->err = g_error_new (self->gquark_err_tag, ENODATA,
        "%s: failed to wrap the raw data of recieved message in GstMemory: %s",
        __func__, g_strerror (ENODATA));
    return TRUE;
  }

  mqtt_msg_hdr = _extract_mqtt_msg_hdr_from (recieved_mem, &hdr_mem,
      &hdr_map_info);
  if (!mqtt_msg_hdr) {
    self->err = g_error_new (self->gquark_err_tag, ENODATA,
        "%s: failed to extract header information from recieved message: %s",
        __func__, g_strerror (ENODATA));
    goto ret_unref_recieved_mem;
  }

  buffer = gst_buffer_new ();
  offset = GST_MQTT_LEN_MSG_HDR;
  for (i = 0; i < mqtt_msg_hdr->num_mems; ++i) {
    GstMemory *each_memory;
    int each_size;

    each_size = mqtt_msg_hdr->size_mems[i];
    each_memory = gst_memory_share (recieved_mem, offset, each_size);
    gst_buffer_append_memory (buffer, each_memory);
    offset += each_size;
  }

  g_async_queue_push (self->aqueue, buffer);

  gst_memory_unmap (hdr_mem, &hdr_map_info);
  gst_memory_unref (hdr_mem);

ret_unref_recieved_mem:
  gst_memory_unref (recieved_mem);

  return TRUE;
}

/**
  * @brief A callback invoked when destroying the GstMemory which wrapped the arrived message
  */
static void
cb_memory_wrapped_destroy (void *p)
{
  MQTTAsync_message *msg = p;

  MQTTAsync_freeMessage (&msg);
}

/**
  * @brief A callback invoked when the connection is established
  */
static void
cb_mqtt_on_connect (void *context, MQTTAsync_successData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);
  MQTTAsync_responseOptions opts = MQTTAsync_responseOptions_initializer;
  int ret;

  GST_OBJECT_LOCK (self);
  self->is_connected = TRUE;
  g_cond_signal (&self->gcond);
  GST_OBJECT_UNLOCK (self);

  opts.context = self;
  opts.onSuccess = cb_mqtt_on_subscribe;
  opts.onFailure = cb_mqtt_on_subscribe_failure;

  /** @todo Support QoS option */
  ret = MQTTAsync_subscribe (*self->mqtt_client_handle, self->mqtt_topic, 1,
      &opts);
  if (ret != MQTTASYNC_SUCCESS) {
    g_printerr ("Failed to start subscribe, return code %d\n", ret);
    return;
  }

  GST_OBJECT_LOCK (self);
  self->is_subscribed = TRUE;
  GST_OBJECT_UNLOCK (self);
}

/**
  * @brief A callback invoked when it is failed to connect to the broker
  */
static void
cb_mqtt_on_connect_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);

  GST_OBJECT_LOCK (self);
  self->is_connected = FALSE;
  self->err = g_error_new (self->gquark_err_tag, response->code,
      "%s: failed to connect to the broker: %s", __func__, response->message);
  g_cond_signal (&self->gcond);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief An implementation for the onSuccess callback of MQTTAsync_responseOptions
 */
static void
cb_mqtt_on_subscribe (void *context, MQTTAsync_successData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);

  GST_OBJECT_LOCK (self);
  self->is_subscribed = TRUE;
  g_cond_signal (&self->gcond);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief An implementation for the onFailure callback of MQTTAsync_responseOptions
 */
static void
cb_mqtt_on_subscribe_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);

  GST_OBJECT_LOCK (self);
  self->is_connected = FALSE;
  self->err = g_error_new (self->gquark_err_tag, response->code,
      "%s: failed to subscribe the given topic, %s: %s", __func__,
      self->mqtt_topic, response->message);
  g_cond_signal (&self->gcond);
  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief A utility function to extract header information from a received message
 */
static GstMQTTMessageHdr *
_extract_mqtt_msg_hdr_from (GstMemory * mem, GstMemory ** hdr_mem,
    GstMapInfo * hdr_map_info)
{
  *hdr_mem = gst_memory_share (mem, 0, GST_MQTT_LEN_MSG_HDR);
  g_return_val_if_fail (*hdr_mem != NULL, NULL);

  if (!gst_memory_map (*hdr_mem, hdr_map_info, GST_MAP_READ)) {
    gst_memory_unref (*hdr_mem);
    return NULL;
  }

  return (GstMQTTMessageHdr *) hdr_map_info->data;
}
