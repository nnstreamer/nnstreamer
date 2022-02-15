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
#include <nnstreamer_util.h>

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

  PROP_DEBUG,
  PROP_IS_LIVE,
  PROP_MQTT_CLIENT_ID,
  PROP_MQTT_HOST_ADDRESS,
  PROP_MQTT_HOST_PORT,
  PROP_MQTT_SUB_TOPIC,
  PROP_MQTT_SUB_TIMEOUT,
  PROP_MQTT_OPT_CLEANSESSION,
  PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL,
  PROP_MQTT_QOS,

  PROP_LAST
};

enum
{
  DEFAULT_DEBUG = FALSE,
  DEFAULT_IS_LIVE = TRUE,
  DEFAULT_MQTT_OPT_CLEANSESSION = TRUE,
  DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL = 60,    /* 1 minute */
  DEFAULT_MQTT_SUB_TIMEOUT = 10000000,  /* 10 seconds */
  DEFAULT_MQTT_SUB_TIMEOUT_MIN = 1000000,       /* 1 seconds */
  DEFAULT_MQTT_QOS = 2,         /* Once and one only */
};

static guint8 src_client_id = 0;
static const gchar DEFAULT_MQTT_HOST_ADDRESS[] = "tcp://localhost";
static const gchar DEFAULT_MQTT_HOST_PORT[] = "1883";
static const gchar TAG_ERR_MQTTSRC[] = "ERROR: MQTTSrc";
static const gchar DEFAULT_MQTT_CLIENT_ID[] =
    "$HOSTNAME_$PID_^[0-9][0-9]?$|^255$";
static const gchar DEFAULT_MQTT_CLIENT_ID_FORMAT[] = "%s_%u_src%u";

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
static gboolean gst_mqtt_src_renegotiate (GstBaseSrc * basesrc);

static void
gst_mqtt_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end);
static gboolean gst_mqtt_src_is_seekable (GstBaseSrc * basesrc);
static GstFlowReturn
gst_mqtt_src_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** buf);
static gboolean gst_mqtt_src_query (GstBaseSrc * basesrc, GstQuery * query);

static gboolean gst_mqtt_src_get_debug (GstMqttSrc * self);
static void gst_mqtt_src_set_debug (GstMqttSrc * self, const gboolean flag);
static gboolean gst_mqtt_src_get_is_live (GstMqttSrc * self);
static void gst_mqtt_src_set_is_live (GstMqttSrc * self, const gboolean flag);
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
static gint gst_mqtt_src_get_mqtt_qos (GstMqttSrc * self);
static void gst_mqtt_src_set_mqtt_qos (GstMqttSrc * self, const gint qos);

static void cb_mqtt_on_connection_lost (void *context, char *cause);
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
static void cb_mqtt_on_unsubscribe (void *context,
    MQTTAsync_successData * response);
static void cb_mqtt_on_unsubscribe_failure (void *context,
    MQTTAsync_failureData * response);

static void cb_memory_wrapped_destroy (void *p);

static GstMQTTMessageHdr *_extract_mqtt_msg_hdr_from (GstMemory * mem,
    GstMemory ** hdr_mem, GstMapInfo * hdr_map_info);
static void _put_timestamp_on_gst_buf (GstMqttSrc * self,
    GstMQTTMessageHdr * hdr, GstBuffer * buf);
static gboolean _subscribe (GstMqttSrc * self);
static gboolean _unsubscribe (GstMqttSrc * self);

/**
 * @brief A utility function to check whether the timestamp marked by _put_timestamp_on_gst_buf () is valid or not
 */
static inline gboolean
_is_gst_buffer_timestamp_valid (GstBuffer * buf)
{
  if (!GST_BUFFER_PTS_IS_VALID (buf) && !GST_BUFFER_DTS_IS_VALID (buf) &&
      !GST_BUFFER_DURATION_IS_VALID (buf))
    return FALSE;
  return TRUE;
}

/** Function defintions */
/**
 * @brief Initialize GstMqttSrc object
 */
static void
gst_mqtt_src_init (GstMqttSrc * self)
{
  MQTTAsync_connectOptions conn_opts = MQTTAsync_connectOptions_initializer;
  MQTTAsync_responseOptions respn_opts = MQTTAsync_responseOptions_initializer;
  GstBaseSrc *basesrc = GST_BASE_SRC (self);

  self->gquark_err_tag = g_quark_from_string (TAG_ERR_MQTTSRC);

  gst_base_src_set_format (basesrc, GST_FORMAT_TIME);
  gst_base_src_set_async (basesrc, FALSE);

  /** init mqttsrc properties */
  self->mqtt_client_handle = NULL;
  self->debug = DEFAULT_DEBUG;
  self->is_live = DEFAULT_IS_LIVE;
  self->mqtt_client_id = g_strdup (DEFAULT_MQTT_CLIENT_ID);
  self->mqtt_host_address = g_strdup (DEFAULT_MQTT_HOST_ADDRESS);
  self->mqtt_host_port = g_strdup (DEFAULT_MQTT_HOST_PORT);
  self->mqtt_topic = NULL;
  self->mqtt_sub_timeout = (gint64) DEFAULT_MQTT_SUB_TIMEOUT;
  self->mqtt_conn_opts = conn_opts;
  self->mqtt_conn_opts.cleansession = DEFAULT_MQTT_OPT_CLEANSESSION;
  self->mqtt_conn_opts.keepAliveInterval = DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL;
  self->mqtt_conn_opts.onSuccess = cb_mqtt_on_connect;
  self->mqtt_conn_opts.onFailure = cb_mqtt_on_connect_failure;
  self->mqtt_conn_opts.context = self;
  self->mqtt_respn_opts = respn_opts;
  self->mqtt_respn_opts.onSuccess = NULL;
  self->mqtt_respn_opts.onFailure = NULL;
  self->mqtt_respn_opts.context = self;
  self->mqtt_qos = DEFAULT_MQTT_QOS;

  /** init private member variables */
  self->err = NULL;
  self->aqueue = g_async_queue_new ();
  g_cond_init (&self->mqtt_src_gcond);
  g_mutex_init (&self->mqtt_src_mutex);
  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_connected = FALSE;
  self->is_subscribed = FALSE;
  self->latency = GST_CLOCK_TIME_NONE;
  g_mutex_unlock (&self->mqtt_src_mutex);
  self->base_time_epoch = GST_CLOCK_TIME_NONE;
  self->caps = NULL;
  self->num_dumped = 0;

  gst_base_src_set_live (basesrc, self->is_live);
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

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, GST_MQTT_ELEM_NAME_SRC, 0,
      "MQTT src");

  gobject_class->set_property = gst_mqtt_src_set_property;
  gobject_class->get_property = gst_mqtt_src_get_property;
  gobject_class->finalize = gst_mqtt_src_class_finalize;

  g_object_class_install_property (gobject_class, PROP_DEBUG,
      g_param_spec_boolean ("debug", "Debug",
          "Produce extra verbose output for debug purpose", DEFAULT_DEBUG,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_IS_LIVE,
      g_param_spec_boolean ("is-live", "Is Live",
          "Synchronize the incoming buffers' timestamp with the current running time",
          DEFAULT_IS_LIVE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
      g_param_spec_string ("sub-topic", "Topic to Subscribe (mandatory)",
          "The topic's name to subscribe", NULL,
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

  g_object_class_install_property (gobject_class, PROP_MQTT_QOS,
      g_param_spec_int ("mqtt-qos", "mqtt QoS level",
          "The QoS level of MQTT.\n"
          "\t\t\t  0: At most once\n"
          "\t\t\t  1: At least once\n"
          "\t\t\t  2: Exactly once\n"
          "\t\t\tsee also: https://www.eclipse.org/paho/files/mqttdoc/MQTTAsync/html/qos.html",
          0, 2, DEFAULT_MQTT_QOS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_mqtt_src_change_state);

  gstbasesrc_class->start = GST_DEBUG_FUNCPTR (gst_mqtt_src_start);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR (gst_mqtt_src_stop);
  gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_caps);
  gstbasesrc_class->get_times = GST_DEBUG_FUNCPTR (gst_mqtt_src_get_times);
  gstbasesrc_class->is_seekable = GST_DEBUG_FUNCPTR (gst_mqtt_src_is_seekable);
  gstbasesrc_class->create = GST_DEBUG_FUNCPTR (gst_mqtt_src_create);
  gstbasesrc_class->query = GST_DEBUG_FUNCPTR (gst_mqtt_src_query);

  gst_element_class_set_static_metadata (gstelement_class,
      "MQTT source", "Source/MQTT",
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
    case PROP_DEBUG:
      gst_mqtt_src_set_debug (self, g_value_get_boolean (value));
      break;
    case PROP_IS_LIVE:
      gst_mqtt_src_set_is_live (self, g_value_get_boolean (value));
      break;
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
    case PROP_MQTT_QOS:
      gst_mqtt_src_set_mqtt_qos (self, g_value_get_int (value));
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
    case PROP_DEBUG:
      g_value_set_boolean (value, gst_mqtt_src_get_debug (self));
      break;
    case PROP_IS_LIVE:
      g_value_set_boolean (value, gst_mqtt_src_get_is_live (self));
      break;
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
    case PROP_MQTT_QOS:
      g_value_set_int (value, gst_mqtt_src_get_mqtt_qos (self));
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
  GstBuffer *remained;

  if (self->mqtt_client_handle) {
    MQTTAsync_destroy (&self->mqtt_client_handle);
    self->mqtt_client_handle = NULL;
  }

  g_free (self->mqtt_client_id);
  g_free (self->mqtt_host_address);
  g_free (self->mqtt_host_port);
  g_free (self->mqtt_topic);
  gst_caps_replace (&self->caps, NULL);

  if (self->err)
    g_error_free (self->err);

  while ((remained = g_async_queue_try_pop (self->aqueue))) {
    gst_buffer_unref (remained);
  }
  g_clear_pointer (&self->aqueue, g_async_queue_unref);

  g_mutex_clear (&self->mqtt_src_mutex);
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
  gboolean no_preroll = FALSE;
  GstClock *elem_clock;
  GstClockTime base_time;
  GstClockTime cur_time;
  GstClockTimeDiff diff;

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
      /* Regardless of the 'is-live''s value, prerolling is not supported */
      no_preroll = TRUE;
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      GST_INFO_OBJECT (self, "GST_STATE_CHANGE_PAUSED_TO_PLAYING");
      self->base_time_epoch = GST_CLOCK_TIME_NONE;
      elem_clock = gst_element_get_clock (element);
      if (!elem_clock)
        break;
      base_time = gst_element_get_base_time (element);
      cur_time = gst_clock_get_time (elem_clock);
      gst_object_unref (elem_clock);
      diff = GST_CLOCK_DIFF (base_time, cur_time);
      self->base_time_epoch =
          g_get_real_time () * GST_US_TO_NS_MULTIPLIER - diff;

      /** This handles the case when the state is changed to PLAYING again */
      if (GST_BASE_SRC_IS_STARTED (GST_BASE_SRC (self)) &&
          (self->is_connected == FALSE)) {
        int conn = MQTTAsync_reconnect (self->mqtt_client_handle);

        if (conn != MQTTASYNC_SUCCESS) {
          GST_ERROR_OBJECT (self, "Failed to re-subscribe to %s",
              self->mqtt_topic);

          return GST_STATE_CHANGE_FAILURE;
        }
      }
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      if (self->is_subscribed && !_unsubscribe (self)) {
        GST_ERROR_OBJECT (self, "Cannot unsubscribe to %s", self->mqtt_topic);
      }
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

  if (no_preroll && ret == GST_STATE_CHANGE_SUCCESS)
    ret = GST_STATE_CHANGE_NO_PREROLL;

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
  gint64 end_time;

  if (!g_strcmp0 (DEFAULT_MQTT_CLIENT_ID, self->mqtt_client_id)) {
    g_free (self->mqtt_client_id);
    self->mqtt_client_id = g_strdup_printf (DEFAULT_MQTT_CLIENT_ID_FORMAT,
        g_get_host_name (), getpid (), src_client_id++);
  }

  /**
   * @todo Support other persistence mechanisms
   *    MQTTCLIENT_PERSISTENCE_NONE: A memory-based persistence mechanism
   *    MQTTCLIENT_PERSISTENCE_DEFAULT: The default file system-based
   *                                    persistence mechanism
   *    MQTTCLIENT_PERSISTENCE_USER: An application-specific persistence
   *                                 mechanism
   */
  ret = MQTTAsync_create (&self->mqtt_client_handle, haddr,
      self->mqtt_client_id, MQTTCLIENT_PERSISTENCE_NONE, NULL);
  g_free (haddr);
  if (ret != MQTTASYNC_SUCCESS)
    return FALSE;

  MQTTAsync_setCallbacks (self->mqtt_client_handle, self,
      cb_mqtt_on_connection_lost, cb_mqtt_on_message_arrived, NULL);

  ret = MQTTAsync_connect (self->mqtt_client_handle, &self->mqtt_conn_opts);
  if (ret != MQTTASYNC_SUCCESS)
    goto error;

  /* Waiting for the connection */
  end_time = g_get_monotonic_time () +
      DEFAULT_MQTT_CONN_TIMEOUT_SEC * G_TIME_SPAN_SECOND;
  g_mutex_lock (&self->mqtt_src_mutex);
  while (!self->is_connected) {
    if (!g_cond_wait_until (&self->mqtt_src_gcond, &self->mqtt_src_mutex,
            end_time)) {
      g_mutex_unlock (&self->mqtt_src_mutex);
      g_critical ("Failed to connect to MQTT broker from mqttsrc."
          "Please check broker is running status or broker host address.");
      goto error;
    }
  }
  g_mutex_unlock (&self->mqtt_src_mutex);
  return TRUE;

error:
  MQTTAsync_destroy (&self->mqtt_client_handle);
  self->mqtt_client_handle = NULL;
  return FALSE;
}

/**
 * @brief Stop mqttsrc, called when state changed ready to null
 */
static gboolean
gst_mqtt_src_stop (GstBaseSrc * basesrc)
{
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);

  /* todo */
  MQTTAsync_disconnect (self->mqtt_client_handle, NULL);
  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_connected = FALSE;
  g_mutex_unlock (&self->mqtt_src_mutex);
  MQTTAsync_destroy (&self->mqtt_client_handle);
  self->mqtt_client_handle = NULL;
  return TRUE;
}

/**
 * @brief Get caps of subclass
 */
static GstCaps *
gst_mqtt_src_get_caps (GstBaseSrc * basesrc, GstCaps * filter)
{
  GstPad *pad = GST_BASE_SRC_PAD (basesrc);
  GstCaps *cur_caps = gst_pad_get_current_caps (pad);
  GstCaps *caps = gst_caps_new_any ();
  UNUSED (filter);

  if (cur_caps) {
    GstCaps *intersection =
        gst_caps_intersect_full (cur_caps, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (cur_caps);
    gst_caps_unref (caps);
    caps = intersection;
  }

  return caps;
}

/**
 * @brief Do negotiation procedure again if it needed
 */
static gboolean
gst_mqtt_src_renegotiate (GstBaseSrc * basesrc)
{
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);
  GstCaps *caps = NULL;
  GstCaps *peercaps = NULL;
  GstCaps *thiscaps;
  gboolean result = FALSE;
  GstCaps *fixed_caps = NULL;

  if (self->caps == NULL || gst_caps_is_any (self->caps))
    goto no_nego_needed;

  thiscaps = gst_pad_query_caps (GST_BASE_SRC_PAD (basesrc), NULL);
  if (thiscaps && gst_caps_is_equal (self->caps, thiscaps)) {
    gst_caps_unref (thiscaps);
    goto no_nego_needed;
  }

  peercaps = gst_pad_peer_query_caps (GST_BASE_SRC_PAD (basesrc), self->caps);
  if (peercaps && !gst_caps_is_empty (peercaps)) {
    caps = gst_caps_ref (peercaps);
    if (peercaps != self->caps)
      gst_caps_unref (peercaps);
  } else {
    caps = gst_caps_ref (self->caps);
  }

  if (caps && !gst_caps_is_empty (caps)) {
    if (gst_caps_is_any (caps)) {
      result = TRUE;
    } else {
      fixed_caps = gst_caps_fixate (caps);
      if (fixed_caps && gst_caps_is_fixed (fixed_caps)) {
        result = gst_base_src_set_caps (basesrc, fixed_caps);
        if (peercaps == self->caps)
          gst_caps_unref (fixed_caps);
      }
    }
    gst_caps_unref (caps);
  } else {
    result = FALSE;
    if (caps && gst_caps_is_empty (caps))
      gst_caps_unref (caps);
  }

  if (thiscaps)
    gst_caps_unref (thiscaps);

  return result;

no_nego_needed:
  {
    GST_DEBUG_OBJECT (self, "no negotiation needed");

    return TRUE;
  }
}

/**
 * @brief Return the time information of the given buffer
 */
static void
gst_mqtt_src_get_times (GstBaseSrc * basesrc, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
  GstClockTime sync_ts;
  GstClockTime duration;
  UNUSED (basesrc);

  sync_ts = GST_BUFFER_DTS (buffer);
  duration = GST_BUFFER_DURATION (buffer);

  if (!GST_CLOCK_TIME_IS_VALID (sync_ts))
    sync_ts = GST_BUFFER_PTS (buffer);

  if (GST_CLOCK_TIME_IS_VALID (sync_ts)) {
    *start = sync_ts;
    if (GST_CLOCK_TIME_IS_VALID (duration)) {
      *end = sync_ts + duration;
    }
  }
}

/**
 * @brief Check if source supports seeking
 * @note Seeking is not supported since this element handles live subscription data.
 */
static gboolean
gst_mqtt_src_is_seekable (GstBaseSrc * basesrc)
{
  UNUSED (basesrc);
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
  UNUSED (offset);
  UNUSED (size);

  g_mutex_lock (&self->mqtt_src_mutex);
  while ((!self->is_connected) || (!self->is_subscribed)) {
    gint64 end_time = g_get_monotonic_time () + G_TIME_SPAN_SECOND;

    g_cond_wait_until (&self->mqtt_src_gcond, &self->mqtt_src_mutex, end_time);
    if (self->err) {
      g_mutex_unlock (&self->mqtt_src_mutex);
      goto ret_flow_err;
    }
  }
  g_mutex_unlock (&self->mqtt_src_mutex);

  while (elapsed > 0) {
    /** @todo DEFAULT_MQTT_SUB_TIMEOUT_MIN is too long */
    *buf = g_async_queue_timeout_pop (self->aqueue,
        DEFAULT_MQTT_SUB_TIMEOUT_MIN);
    if (*buf) {
      GstClockTime base_time = gst_element_get_base_time (GST_ELEMENT (self));
      GstClockTime ulatency = GST_CLOCK_TIME_NONE;
      GstClock *clock;

      /** This buffer is comming from the past. Drop it */
      if (!_is_gst_buffer_timestamp_valid (*buf)) {
        if (self->debug) {
          GST_DEBUG_OBJECT (self,
              "%s: Dumped the received buffer! (total: %" G_GUINT64_FORMAT ")",
              self->mqtt_topic, ++self->num_dumped);
        }
        elapsed = self->mqtt_sub_timeout;
        gst_buffer_unref (*buf);
        continue;
      }

      /** Update latency */
      clock = gst_element_get_clock (GST_ELEMENT (self));
      if (clock) {
        GstClockTime cur_time = gst_clock_get_time (clock);
        GstClockTime buf_ts = GST_BUFFER_TIMESTAMP (*buf);
        GstClockTimeDiff latency = 0;

        if ((base_time != GST_CLOCK_TIME_NONE) &&
            (cur_time != GST_CLOCK_TIME_NONE) &&
            (buf_ts != GST_CLOCK_TIME_NONE)) {
          GstClockTimeDiff now = GST_CLOCK_DIFF (base_time, cur_time);

          latency = GST_CLOCK_DIFF (buf_ts, (GstClockTime) now);
        }

        if (latency > 0) {
          ulatency = (GstClockTime) latency;

          if (GST_BUFFER_DURATION_IS_VALID (*buf)) {
            GstClockTime duration = GST_BUFFER_DURATION (*buf);

            if (duration >= ulatency) {
              ulatency = GST_CLOCK_TIME_NONE;
            }
          }
        }
        gst_object_unref (clock);
      }

      g_mutex_lock (&self->mqtt_src_mutex);
      self->latency = ulatency;
      g_mutex_unlock (&self->mqtt_src_mutex);
      /**
       * @todo If the difference between new latency and old latency,
       *      gst_element_post_message (GST_ELEMENT_CAST (self),
       *          gst_message_new_latency (GST_OBJECT_CAST (self)));
       *      is needed.
       */
      break;
    } else if (self->err) {
      break;
    }
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
 * @brief An implementation of the GstBaseSrc vmethod that handles queries
 */
static gboolean
gst_mqtt_src_query (GstBaseSrc * basesrc, GstQuery * query)
{
  GstQueryType type = GST_QUERY_TYPE (query);
  GstMqttSrc *self = GST_MQTT_SRC (basesrc);
  gboolean res = FALSE;

  if (self->debug)
    GST_DEBUG_OBJECT (self, "Got %s event", gst_query_type_get_name (type));

  switch (type) {
    case GST_QUERY_LATENCY:{
      GstClockTime min_latency = 0;
      GstClockTime max_latency = GST_CLOCK_TIME_NONE;

      g_mutex_lock (&self->mqtt_src_mutex);
      if (self->latency != GST_CLOCK_TIME_NONE) {
        min_latency = self->latency;
      }
      g_mutex_unlock (&self->mqtt_src_mutex);

      if (self->debug) {
        GST_DEBUG_OBJECT (self,
            "Reporting latency min %" GST_TIME_FORMAT ", max %" GST_TIME_FORMAT,
            GST_TIME_ARGS (min_latency), GST_TIME_ARGS (max_latency));
      }
      /**
       * @brief The second argument of gst_query_set_latency should be always
       *        TRUE.
       */
      gst_query_set_latency (query, TRUE, min_latency, max_latency);

      res = TRUE;
      break;
    }
    default:{
      res = GST_BASE_SRC_CLASS (parent_class)->query (basesrc, query);
    }
  }

  return res;
}

/**
 * @brief Getter for the 'debug' property.
 */
static gboolean
gst_mqtt_src_get_debug (GstMqttSrc * self)
{
  return self->debug;
}

/**
 * @brief Setter for the 'debug' property.
 */
static void
gst_mqtt_src_set_debug (GstMqttSrc * self, const gboolean flag)
{
  self->debug = flag;
}

/**
 * @brief Getter for the 'is-live' property.
 */
static gboolean
gst_mqtt_src_get_is_live (GstMqttSrc * self)
{
  return self->is_live;
}

/**
 * @brief Setter for the 'is-live' property.
 */
static void
gst_mqtt_src_set_is_live (GstMqttSrc * self, const gboolean flag)
{
  self->is_live = flag;
  gst_base_src_set_live (GST_BASE_SRC (self), self->is_live);
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
  g_free (self->mqtt_client_id);
  self->mqtt_client_id = g_strdup (id);
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
  self->mqtt_host_address = g_strdup (addr);
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
  self->mqtt_host_port = g_strdup (port);
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
  self->mqtt_sub_timeout = t;
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
  g_free (self->mqtt_topic);
  self->mqtt_topic = g_strdup (topic);
}

/**
 * @brief Getter for the 'cleansession' property.
 */
static gboolean
gst_mqtt_src_get_opt_cleansession (GstMqttSrc * self)
{
  return self->mqtt_conn_opts.cleansession;
}

/**
 * @brief Setter for the 'cleansession' property.
 */
static void
gst_mqtt_src_set_opt_cleansession (GstMqttSrc * self, const gboolean val)
{
  self->mqtt_conn_opts.cleansession = val;
}

/**
 * @brief Getter for the 'keep-alive-interval' property
 */
static gint
gst_mqtt_src_get_opt_keep_alive_interval (GstMqttSrc * self)
{
  return self->mqtt_conn_opts.keepAliveInterval;
}

/**
 * @brief Setter for the 'keep-alive-interval' property
 */
static void
gst_mqtt_src_set_opt_keep_alive_interval (GstMqttSrc * self, const gint num)
{
  self->mqtt_conn_opts.keepAliveInterval = num;
}

/**
 * @brief Getter for the 'mqtt-qos' property
 */
static gint
gst_mqtt_src_get_mqtt_qos (GstMqttSrc * self)
{
  return self->mqtt_qos;
}

/**
 * @brief Setter for the 'mqtt-qos' property
 */
static void
gst_mqtt_src_set_mqtt_qos (GstMqttSrc * self, const gint qos)
{
  self->mqtt_qos = qos;
}

/**
  * @brief A callback to handle the connection lost to the broker
  */
static void
cb_mqtt_on_connection_lost (void *context, char *cause)
{
  GstMqttSrc *self = GST_MQTT_SRC_CAST (context);
  UNUSED (cause);

  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_connected = FALSE;
  self->is_subscribed = FALSE;
  g_cond_broadcast (&self->mqtt_src_gcond);
  if (!self->err) {
    self->err = g_error_new (self->gquark_err_tag, EHOSTDOWN,
        "Connection to the host (broker) has been lost: %s \n"
        "\t\tfor detail, please check the log message of the broker",
        g_strerror (EHOSTDOWN));
  }
  g_mutex_unlock (&self->mqtt_src_mutex);
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
  GstMemory *received_mem;
  GstMemory *hdr_mem;
  GstBuffer *buffer;
  GstBaseSrc *basesrc;
  GstMqttSrc *self;
  GstClock *clock;
  gsize offset;
  guint i;
  UNUSED (topic_name);
  UNUSED (topic_len);

  self = GST_MQTT_SRC_CAST (context);
  g_mutex_lock (&self->mqtt_src_mutex);
  if (!self->is_subscribed) {
    g_mutex_unlock (&self->mqtt_src_mutex);

    return TRUE;
  }
  g_mutex_unlock (&self->mqtt_src_mutex);

  basesrc = GST_BASE_SRC (self);
  clock = gst_element_get_clock (GST_ELEMENT (self));
  received_mem = gst_memory_new_wrapped (0, data, size, 0, size, message,
      (GDestroyNotify) cb_memory_wrapped_destroy);
  if (!received_mem) {
    if (!self->err) {
      self->err = g_error_new (self->gquark_err_tag, ENODATA,
          "%s: failed to wrap the raw data of received message in GstMemory: %s",
          __func__, g_strerror (ENODATA));
    }
    return TRUE;
  }

  mqtt_msg_hdr = _extract_mqtt_msg_hdr_from (received_mem, &hdr_mem,
      &hdr_map_info);
  if (!mqtt_msg_hdr) {
    if (!self->err) {
      self->err = g_error_new (self->gquark_err_tag, ENODATA,
          "%s: failed to extract header information from received message: %s",
          __func__, g_strerror (ENODATA));
    }
    goto ret_unref_received_mem;
  }

  if (!self->caps) {
    self->caps = gst_caps_from_string (mqtt_msg_hdr->gst_caps_str);
    gst_mqtt_src_renegotiate (basesrc);
  } else {
    GstCaps *recv_caps = gst_caps_from_string (mqtt_msg_hdr->gst_caps_str);

    if (recv_caps && !gst_caps_is_equal (self->caps, recv_caps)) {
      gst_caps_replace (&self->caps, recv_caps);
      gst_mqtt_src_renegotiate (basesrc);
    } else {
      gst_caps_replace (&recv_caps, NULL);
    }
  }

  buffer = gst_buffer_new ();
  offset = GST_MQTT_LEN_MSG_HDR;
  for (i = 0; i < mqtt_msg_hdr->num_mems; ++i) {
    GstMemory *each_memory;
    int each_size;

    each_size = mqtt_msg_hdr->size_mems[i];
    each_memory = gst_memory_share (received_mem, offset, each_size);
    gst_buffer_append_memory (buffer, each_memory);
    offset += each_size;
  }

  /** Timestamp synchronization */
  if (self->debug) {
    GstClockTime base_time = gst_element_get_base_time (GST_ELEMENT (self));

    if (clock) {
      GST_DEBUG_OBJECT (self,
          "A message has been arrived at %" GST_TIME_FORMAT
          " and queue length is %d",
          GST_TIME_ARGS (gst_clock_get_time (clock) - base_time),
          g_async_queue_length (self->aqueue));

      gst_object_unref (clock);
    }
  }
  _put_timestamp_on_gst_buf (self, mqtt_msg_hdr, buffer);
  g_async_queue_push (self->aqueue, buffer);

  gst_memory_unmap (hdr_mem, &hdr_map_info);
  gst_memory_unref (hdr_mem);

ret_unref_received_mem:
  gst_memory_unref (received_mem);

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
  GstBaseSrc *basesrc = GST_BASE_SRC (self);
  int ret;
  UNUSED (response);

  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_connected = TRUE;
  g_cond_broadcast (&self->mqtt_src_gcond);
  g_mutex_unlock (&self->mqtt_src_mutex);

  /** GstFlowReturn is an enum type. It is possible to use int here */
  if (gst_base_src_is_async (basesrc) &&
      (ret = gst_base_src_start_wait (basesrc)) != GST_FLOW_OK) {
    g_mutex_lock (&self->mqtt_src_mutex);
    self->err = g_error_new (self->gquark_err_tag, ret,
        "%s: the virtual method, start (), in the GstBaseSrc class fails with return code %d",
        __func__, ret);
    g_cond_broadcast (&self->mqtt_src_gcond);
    g_mutex_unlock (&self->mqtt_src_mutex);
    return;
  }

  if (!_subscribe (self)) {
    GST_ERROR_OBJECT (self, "Failed to subscribe to %s", self->mqtt_topic);
  }
}

/**
  * @brief A callback invoked when it is failed to connect to the broker
  */
static void
cb_mqtt_on_connect_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);

  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_connected = FALSE;

  if (!self->err) {
    self->err = g_error_new (self->gquark_err_tag, response->code,
        "%s: failed to connect to the broker: %s", __func__, response->message);
  }
  g_cond_broadcast (&self->mqtt_src_gcond);
  g_mutex_unlock (&self->mqtt_src_mutex);
}

/**
 * @brief MQTTAsync_responseOptions's onSuccess callback for MQTTAsync_subscribe ()
 */
static void
cb_mqtt_on_subscribe (void *context, MQTTAsync_successData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);
  UNUSED (response);

  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_subscribed = TRUE;
  g_cond_broadcast (&self->mqtt_src_gcond);
  g_mutex_unlock (&self->mqtt_src_mutex);
}

/**
 * @brief MQTTAsync_responseOptions's onFailure callback for MQTTAsync_subscribe ()
 */
static void
cb_mqtt_on_subscribe_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);

  g_mutex_lock (&self->mqtt_src_mutex);
  if (!self->err) {
    self->err = g_error_new (self->gquark_err_tag, response->code,
        "%s: failed to subscribe the given topic, %s: %s", __func__,
        self->mqtt_topic, response->message);
  }
  g_cond_broadcast (&self->mqtt_src_gcond);
  g_mutex_unlock (&self->mqtt_src_mutex);
}

/**
 * @brief MQTTAsync_responseOptions's onSuccess callback for MQTTAsync_unsubscribe ()
 */
static void
cb_mqtt_on_unsubscribe (void *context, MQTTAsync_successData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);
  UNUSED (response);

  g_mutex_lock (&self->mqtt_src_mutex);
  self->is_subscribed = FALSE;
  g_cond_broadcast (&self->mqtt_src_gcond);
  g_mutex_unlock (&self->mqtt_src_mutex);
}

/**
 * @brief MQTTAsync_responseOptions's onFailure callback for MQTTAsync_unsubscribe ()
 */
static void
cb_mqtt_on_unsubscribe_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSrc *self = GST_MQTT_SRC (context);

  g_mutex_lock (&self->mqtt_src_mutex);
  if (!self->err) {
    self->err = g_error_new (self->gquark_err_tag, response->code,
        "%s: failed to unsubscribe the given topic, %s: %s", __func__,
        self->mqtt_topic, response->message);
  }
  g_cond_broadcast (&self->mqtt_src_gcond);
  g_mutex_unlock (&self->mqtt_src_mutex);
}

/**
 * @brief A helper function to properly invoke MQTTAsync_subscribe ()
 */
static gboolean
_subscribe (GstMqttSrc * self)
{
  MQTTAsync_responseOptions opts = self->mqtt_respn_opts;
  int mqttasync_ret;

  opts.onSuccess = cb_mqtt_on_subscribe;
  opts.onFailure = cb_mqtt_on_subscribe_failure;
  opts.subscribeOptions.retainHandling = 1;

  mqttasync_ret = MQTTAsync_subscribe (self->mqtt_client_handle,
      self->mqtt_topic, self->mqtt_qos, &opts);
  if (mqttasync_ret != MQTTASYNC_SUCCESS)
    return FALSE;
  return TRUE;
}

/**
 * @brief A wrapper function that calls MQTTAsync_unsubscribe ()
 */
static gboolean
_unsubscribe (GstMqttSrc * self)
{
  MQTTAsync_responseOptions opts = self->mqtt_respn_opts;
  int mqttasync_ret;

  opts.onSuccess = cb_mqtt_on_unsubscribe;
  opts.onFailure = cb_mqtt_on_unsubscribe_failure;

  mqttasync_ret = MQTTAsync_unsubscribe (self->mqtt_client_handle,
      self->mqtt_topic, &opts);
  if (mqttasync_ret != MQTTASYNC_SUCCESS)
    return FALSE;
  return TRUE;
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

/**
  * @brief A utility function to put the timestamp information
  *        onto a GstBuffer-typed buffer using the given packet header
  */
static void
_put_timestamp_on_gst_buf (GstMqttSrc * self, GstMQTTMessageHdr * hdr,
    GstBuffer * buf)
{
  gint64 diff_base_epoch = hdr->base_time_epoch - self->base_time_epoch;

  buf->pts = GST_CLOCK_TIME_NONE;
  buf->dts = GST_CLOCK_TIME_NONE;
  buf->duration = GST_CLOCK_TIME_NONE;

  if (hdr->sent_time_epoch < self->base_time_epoch)
    return;

  if (((GstClockTimeDiff) hdr->pts + diff_base_epoch) < 0)
    return;

  if (hdr->pts != GST_CLOCK_TIME_NONE) {
    buf->pts = hdr->pts + diff_base_epoch;
  }

  if (hdr->dts != GST_CLOCK_TIME_NONE) {
    buf->dts = hdr->dts + diff_base_epoch;
  }

  buf->duration = hdr->duration;

  if (self->debug) {
    GstClockTime base_time = gst_element_get_base_time (GST_ELEMENT (self));
    GstClock *clock;

    clock = gst_element_get_clock (GST_ELEMENT (self));

    if (clock) {
      GST_DEBUG_OBJECT (self,
          "%s diff %" GST_STIME_FORMAT " now %" GST_TIME_FORMAT " ts (%"
          GST_TIME_FORMAT " -> %" GST_TIME_FORMAT ")", self->mqtt_topic,
          GST_STIME_ARGS (diff_base_epoch),
          GST_TIME_ARGS (gst_clock_get_time (clock) - base_time),
          GST_TIME_ARGS (hdr->pts), GST_TIME_ARGS (buf->pts));

      gst_object_unref (clock);
    }
  }
}
