/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsink.c
 * @date    01 Apr 2021
 * @brief   Publish incoming data streams as a MQTT topic
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>

#ifdef G_OS_WIN32
#include <process.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

#include <gst/base/gstbasesink.h>
#include <MQTTAsync.h>
#include <nnstreamer_util.h>

#include "mqttsink.h"
#include "ntputil.h"

static GstStaticPadTemplate sink_pad_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

#define gst_mqtt_sink_parent_class parent_class
G_DEFINE_TYPE (GstMqttSink, gst_mqtt_sink, GST_TYPE_BASE_SINK);

GST_DEBUG_CATEGORY_STATIC (gst_mqtt_sink_debug);
#define GST_CAT_DEFAULT gst_mqtt_sink_debug

enum
{
  PROP_0,

  PROP_DEBUG,
  PROP_MQTT_CLIENT_ID,
  PROP_MQTT_HOST_ADDRESS,
  PROP_MQTT_HOST_PORT,
  PROP_MQTT_PUB_TOPIC,
  PROP_MQTT_PUB_WAIT_TIMEOUT,
  PROP_MQTT_OPT_CLEANSESSION,
  PROP_MQTT_OPT_KEEP_ALIVE_INTERVAL,
  PROP_NUM_BUFFERS,
  PROP_MAX_MSG_BUF_SIZE,
  PROP_MQTT_QOS,
  PROP_MQTT_NTP_SYNC,
  PROP_MQTT_NTP_SRVS,

  PROP_LAST
};

enum
{
  DEFAULT_DEBUG = FALSE,
  DEFAULT_NUM_BUFFERS = -1,
  DEFAULT_QOS = TRUE,
  DEFAULT_SYNC = FALSE,
  DEFAULT_MQTT_OPT_CLEANSESSION = TRUE,
  DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL = 60,    /* 1 minute */
  DEFAULT_MQTT_DISCONNECT_TIMEOUT = G_TIME_SPAN_SECOND * 3,     /* 3 secs */
  DEFAULT_MQTT_PUB_WAIT_TIMEOUT = 1,    /* 1 secs */
  DEFAULT_MAX_MSG_BUF_SIZE = 0, /* Buffer size is not fixed */
  DEFAULT_MQTT_QOS = 0,         /* fire and forget */
  DEFAULT_MQTT_NTP_SYNC = FALSE,
  MAX_LEN_PROP_NTP_SRVS = 4096,
};

static guint8 sink_client_id = 0;
static const gchar DEFAULT_MQTT_HOST_ADDRESS[] = "127.0.0.1";
static const gchar DEFAULT_MQTT_HOST_PORT[] = "1883";
static const gchar TAG_ERR_MQTTSINK[] = "ERROR: MQTTSink";
static const gchar DEFAULT_MQTT_CLIENT_ID[] = "$HOST_$PID_^[0-9][0-9]?$|^255$";
static const gchar DEFAULT_MQTT_CLIENT_ID_FORMAT[] = "%s_%u_sink%u";
static const gchar DEFAULT_MQTT_PUB_TOPIC[] = "$client-id/topic";
static const gchar DEFAULT_MQTT_PUB_TOPIC_FORMAT[] = "%s/topic";
static const gchar DEFAULT_MQTT_NTP_SERVERS[] = "pool.ntp.org:123";

/**
 * @brief The NTP servers parsed from ntp-srvs, shared by reference so that
 *        replacing them cannot free a list get_epoch_func () is reading.
 */
struct _GstMqttNtpServers
{
  gint ref_count;
  guint num;
  gchar **hnames;
  guint16 *ports;
};

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
static gboolean gst_mqtt_sink_set_caps (GstBaseSink * basesink, GstCaps * caps);

static gboolean gst_mqtt_sink_get_debug (GstMqttSink * self);
static void gst_mqtt_sink_set_debug (GstMqttSink * self, const gboolean flag);
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

static gsize gst_mqtt_sink_get_max_msg_buf_size (GstMqttSink * self);
static void gst_mqtt_sink_set_max_msg_buf_size (GstMqttSink * self,
    const gsize size);
static gint gst_mqtt_sink_get_num_buffers (GstMqttSink * self);
static void gst_mqtt_sink_set_num_buffers (GstMqttSink * self, const gint num);
static gint gst_mqtt_sink_get_mqtt_qos (GstMqttSink * self);
static void gst_mqtt_sink_set_mqtt_qos (GstMqttSink * self, const gint qos);
static gboolean gst_mqtt_sink_get_mqtt_ntp_sync (GstMqttSink * self);
static void gst_mqtt_sink_set_mqtt_ntp_sync (GstMqttSink * self,
    const gboolean flag);
static gchar *gst_mqtt_sink_get_mqtt_ntp_srvs (GstMqttSink * self);
static void gst_mqtt_sink_set_mqtt_ntp_srvs (GstMqttSink * self,
    const gchar * pairs);
static GstMqttNtpServers *gst_mqtt_ntp_servers_new (const gchar * pairs);
static void gst_mqtt_ntp_servers_unref (GstMqttNtpServers * srvs);
static int64_t gst_mqtt_sink_get_epoch (GstMqttSink * self);
static gchar *gst_mqtt_sink_dup_pub_topic (GstMqttSink * self);

static void cb_mqtt_on_connect (void *context,
    MQTTAsync_successData * response);
static void cb_mqtt_on_connect_failure (void *context,
    MQTTAsync_failureData * response);
static void cb_mqtt_on_disconnect (void *context,
    MQTTAsync_successData * response);
static void cb_mqtt_on_disconnect_failure (void *context,
    MQTTAsync_failureData * response);
static void cb_mqtt_on_delivery_complete (void *context, MQTTAsync_token token);
static void cb_mqtt_on_connection_lost (void *context, char *cause);
static int cb_mqtt_on_message_arrived (void *context, char *topicName,
    int topicLen, MQTTAsync_message * message);
static void cb_mqtt_on_send_success (void *context,
    MQTTAsync_successData * response);
static void cb_mqtt_on_send_failure (void *context,
    MQTTAsync_failureData * response);

/**
 * @brief Initialize GstMqttSink object
 */
static void
gst_mqtt_sink_init (GstMqttSink * self)
{
  GstBaseSink *basesink = GST_BASE_SINK (self);
  MQTTAsync_connectOptions conn_opts = MQTTAsync_connectOptions_initializer;
  MQTTAsync_responseOptions respn_opts = MQTTAsync_responseOptions_initializer;

  /** init MQTT related variables */
  self->mqtt_client_handle = NULL;
  self->mqtt_conn_opts = conn_opts;
  self->mqtt_conn_opts.onSuccess = cb_mqtt_on_connect;
  self->mqtt_conn_opts.onFailure = cb_mqtt_on_connect_failure;
  self->mqtt_conn_opts.context = self;
  self->mqtt_respn_opts = respn_opts;
  self->mqtt_respn_opts.onSuccess = cb_mqtt_on_send_success;
  self->mqtt_respn_opts.onFailure = cb_mqtt_on_send_failure;
  self->mqtt_respn_opts.context = self;

  /** init private variables */
  self->mqtt_sink_state = SINK_INITIALIZING;
  self->err = NULL;
  self->gquark_err_tag = g_quark_from_string (TAG_ERR_MQTTSINK);
  g_mutex_init (&self->mqtt_sink_mutex);
  g_cond_init (&self->mqtt_sink_gcond);
  self->mqtt_msg_buf = NULL;
  self->mqtt_msg_buf_size = 0;
  self->is_static_sized_buf = FALSE;
  memset (&self->mqtt_msg_hdr, 0x0, sizeof (self->mqtt_msg_hdr));
  self->base_time_epoch = GST_CLOCK_TIME_NONE;
  self->in_caps = NULL;

  /** init mqttsink properties */
  self->debug = DEFAULT_DEBUG;
  self->num_buffers = DEFAULT_NUM_BUFFERS;
  self->max_msg_buf_size = DEFAULT_MAX_MSG_BUF_SIZE;
  self->mqtt_client_id = g_strdup (DEFAULT_MQTT_CLIENT_ID);
  self->mqtt_host_address = g_strdup (DEFAULT_MQTT_HOST_ADDRESS);
  self->mqtt_host_port = g_strdup (DEFAULT_MQTT_HOST_PORT);
  self->mqtt_topic = g_strdup (DEFAULT_MQTT_PUB_TOPIC);
  self->mqtt_pub_wait_timeout = DEFAULT_MQTT_PUB_WAIT_TIMEOUT;
  self->mqtt_conn_opts.cleansession = DEFAULT_MQTT_OPT_CLEANSESSION;
  self->mqtt_conn_opts.keepAliveInterval = DEFAULT_MQTT_OPT_KEEP_ALIVE_INTERVAL;
  self->mqtt_qos = DEFAULT_MQTT_QOS;
  self->mqtt_ntp_sync = DEFAULT_MQTT_NTP_SYNC;
  self->mqtt_ntp_srvs = g_strdup (DEFAULT_MQTT_NTP_SERVERS);
  self->mqtt_ntp_servers = NULL;
  self->get_epoch_func = default_mqtt_get_unix_epoch;
  self->is_connected = FALSE;

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

  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT, GST_MQTT_ELEM_NAME_SINK, 0,
      "MQTT sink");

  gobject_class->set_property = gst_mqtt_sink_set_property;
  gobject_class->get_property = gst_mqtt_sink_get_property;
  gobject_class->finalize = gst_mqtt_sink_class_finalize;

  g_object_class_install_property (gobject_class, PROP_DEBUG,
      g_param_spec_boolean ("debug", "Debug",
          "Produce extra verbose output for debug purpose", DEFAULT_DEBUG,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_CLIENT_ID,
      g_param_spec_string ("client-id", "Client ID",
          "The client identifier passed to the server (broker).", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_HOST_ADDRESS,
      g_param_spec_string ("host", "Host", "Host (broker) to connect to",
          DEFAULT_MQTT_HOST_ADDRESS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_HOST_PORT,
      g_param_spec_string ("port", "Port",
          "Network port of host (broker) to connect to", DEFAULT_MQTT_HOST_PORT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_NTP_SYNC,
      g_param_spec_boolean ("ntp-sync", "NTP Synchronization",
          "Synchronize received streams to the NTP clock",
          DEFAULT_MQTT_NTP_SYNC, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_NTP_SRVS,
      g_param_spec_string ("ntp-srvs", "NTP Server Host Name and Port Pairs",
          "NTP Servers' HOST_NAME:PORT pairs to use (valid only if ntp-sync is true)\n"
          "\t\t\tUse ',' to separate each pair if there are more pairs than one\n"
          "\t\t\tAn empty or unset value clears the list, and the NTP helper then "
          "uses its own default server",
          DEFAULT_MQTT_NTP_SERVERS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_PUB_TOPIC,
      g_param_spec_string ("pub-topic", "Topic to Publish",
          "The topic's name to publish", NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class,
      PROP_MQTT_PUB_WAIT_TIMEOUT,
      g_param_spec_ulong ("pub-wait-timeout", "Timeout for Publish a message",
          "Timeout for execution of the main thread with completed publication of a message",
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

  g_object_class_install_property (gobject_class, PROP_MAX_MSG_BUF_SIZE,
      g_param_spec_ulong ("max-buffer-size",
          "The maximum size of a message buffer",
          "The maximum size in bytes of a message buffer (0 = dynamic buffer size)",
          0, G_MAXULONG, DEFAULT_MAX_MSG_BUF_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NUM_BUFFERS,
      g_param_spec_int ("num-buffers", "Num Buffers",
          "Number of (remaining) buffers to accept until sending EOS event (-1 = no limit)",
          -1, G_MAXINT32, DEFAULT_NUM_BUFFERS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MQTT_QOS,
      g_param_spec_int ("mqtt-qos", "mqtt QoS level",
          "The QoS level of MQTT.\n"
          "\t\t\t  0: At most once\n"
          "\t\t\t  1: At least once\n"
          "\t\t\t  2: Exactly once\n"
          "\t\t\tsee also: https://www.eclipse.org/paho/files/mqttdoc/MQTTAsync/html/qos.html",
          0, 2, DEFAULT_MQTT_QOS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gstelement_class->change_state = gst_mqtt_sink_change_state;

  gstbasesink_class->start = GST_DEBUG_FUNCPTR (gst_mqtt_sink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR (gst_mqtt_sink_stop);
  gstbasesink_class->query = GST_DEBUG_FUNCPTR (gst_mqtt_sink_query);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR (gst_mqtt_sink_render);
  gstbasesink_class->render_list =
      GST_DEBUG_FUNCPTR (gst_mqtt_sink_render_list);
  gstbasesink_class->event = GST_DEBUG_FUNCPTR (gst_mqtt_sink_event);
  gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR (gst_mqtt_sink_set_caps);

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
    case PROP_DEBUG:
      gst_mqtt_sink_set_debug (self, g_value_get_boolean (value));
      break;
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
    case PROP_MAX_MSG_BUF_SIZE:
      gst_mqtt_sink_set_max_msg_buf_size (self, g_value_get_ulong (value));
      break;
    case PROP_NUM_BUFFERS:
      gst_mqtt_sink_set_num_buffers (self, g_value_get_int (value));
      break;
    case PROP_MQTT_QOS:
      gst_mqtt_sink_set_mqtt_qos (self, g_value_get_int (value));
      break;
    case PROP_MQTT_NTP_SYNC:
      gst_mqtt_sink_set_mqtt_ntp_sync (self, g_value_get_boolean (value));
      break;
    case PROP_MQTT_NTP_SRVS:
      gst_mqtt_sink_set_mqtt_ntp_srvs (self, g_value_get_string (value));
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
    case PROP_DEBUG:
      g_value_set_boolean (value, gst_mqtt_sink_get_debug (self));
      break;
    case PROP_MQTT_CLIENT_ID:
      GST_OBJECT_LOCK (self);
      g_value_set_string (value, gst_mqtt_sink_get_client_id (self));
      GST_OBJECT_UNLOCK (self);
      break;
    case PROP_MQTT_HOST_ADDRESS:
      GST_OBJECT_LOCK (self);
      g_value_set_string (value, gst_mqtt_sink_get_host_address (self));
      GST_OBJECT_UNLOCK (self);
      break;
    case PROP_MQTT_HOST_PORT:
      GST_OBJECT_LOCK (self);
      g_value_set_string (value, gst_mqtt_sink_get_host_port (self));
      GST_OBJECT_UNLOCK (self);
      break;
    case PROP_MQTT_PUB_TOPIC:
      GST_OBJECT_LOCK (self);
      g_value_set_string (value, gst_mqtt_sink_get_pub_topic (self));
      GST_OBJECT_UNLOCK (self);
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
    case PROP_MAX_MSG_BUF_SIZE:
      g_value_set_ulong (value, gst_mqtt_sink_get_max_msg_buf_size (self));
      break;
    case PROP_NUM_BUFFERS:
      g_value_set_int (value, gst_mqtt_sink_get_num_buffers (self));
      break;
    case PROP_MQTT_QOS:
      g_value_set_int (value, gst_mqtt_sink_get_mqtt_qos (self));
      break;
    case PROP_MQTT_NTP_SYNC:
      g_value_set_boolean (value, gst_mqtt_sink_get_mqtt_ntp_sync (self));
      break;
    case PROP_MQTT_NTP_SRVS:
      GST_OBJECT_LOCK (self);
      g_value_set_string (value, gst_mqtt_sink_get_mqtt_ntp_srvs (self));
      GST_OBJECT_UNLOCK (self);
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
  self->mqtt_host_address = NULL;
  g_free (self->mqtt_host_port);
  self->mqtt_host_port = NULL;
  if (self->mqtt_client_handle) {
    MQTTAsync_destroy (&self->mqtt_client_handle);
    self->mqtt_client_handle = NULL;
  }
  g_free (self->mqtt_client_id);
  self->mqtt_client_id = NULL;
  g_free (self->mqtt_msg_buf);
  self->mqtt_msg_buf = NULL;
  g_free (self->mqtt_topic);
  self->mqtt_topic = NULL;
  gst_caps_replace (&self->in_caps, NULL);
  g_free (self->mqtt_ntp_srvs);
  self->mqtt_ntp_srvs = NULL;
  gst_mqtt_ntp_servers_unref (self->mqtt_ntp_servers);
  self->mqtt_ntp_servers = NULL;

  if (self->err)
    g_error_free (self->err);
  g_mutex_clear (&self->mqtt_sink_mutex);
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
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      if (self->mqtt_ntp_sync)
        self->get_epoch_func = ntputil_get_epoch;
      self->base_time_epoch = GST_CLOCK_TIME_NONE;
      elem_clock = gst_element_get_clock (element);
      if (!elem_clock)
        break;
      base_time = gst_element_get_base_time (element);
      cur_time = gst_clock_get_time (elem_clock);
      gst_object_unref (elem_clock);
      diff = GST_CLOCK_DIFF (base_time, cur_time);
      self->base_time_epoch =
          gst_mqtt_sink_get_epoch (self) * GST_US_TO_NS_MULTIPLIER - diff;
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
  gchar *client_id;
  gchar *haddr;
  int ret;
  gint64 end_time;

  GST_OBJECT_LOCK (self);
  haddr = g_strdup_printf ("%s:%s", self->mqtt_host_address,
      self->mqtt_host_port);

  if (!g_strcmp0 (DEFAULT_MQTT_CLIENT_ID, self->mqtt_client_id)) {
    g_free (self->mqtt_client_id);
    self->mqtt_client_id = g_strdup_printf (DEFAULT_MQTT_CLIENT_ID_FORMAT,
        g_get_host_name (), getpid (), sink_client_id++);
  }
  client_id = g_strdup (self->mqtt_client_id);

  if (!g_strcmp0 (DEFAULT_MQTT_PUB_TOPIC, self->mqtt_topic)) {
    g_free (self->mqtt_topic);
    self->mqtt_topic = g_strdup_printf (DEFAULT_MQTT_PUB_TOPIC_FORMAT,
        self->mqtt_client_id);
  }
  GST_OBJECT_UNLOCK (self);

  /**
   * @todo Support other persistence mechanisms
   *    MQTTCLIENT_PERSISTENCE_NONE: A memory-based persistence mechanism
   *    MQTTCLIENT_PERSISTENCE_DEFAULT: The default file system-based
   *                                    persistence mechanism
   *    MQTTCLIENT_PERSISTENCE_USER: An application-specific persistence
   *                                 mechanism
   */
  ret = MQTTAsync_create (&self->mqtt_client_handle, haddr, client_id,
      MQTTCLIENT_PERSISTENCE_NONE, NULL);
  g_free (haddr);
  g_free (client_id);
  if (ret != MQTTASYNC_SUCCESS)
    return FALSE;

  MQTTAsync_setCallbacks (self->mqtt_client_handle, self,
      cb_mqtt_on_connection_lost, cb_mqtt_on_message_arrived,
      cb_mqtt_on_delivery_complete);

  ret = MQTTAsync_connect (self->mqtt_client_handle, &self->mqtt_conn_opts);
  if (ret != MQTTASYNC_SUCCESS) {
    goto error;
  }

  /* Waiting for the connection */
  end_time = g_get_monotonic_time () +
      DEFAULT_MQTT_CONN_TIMEOUT_SEC * G_TIME_SPAN_SECOND;
  g_mutex_lock (&self->mqtt_sink_mutex);
  while (!self->is_connected) {
    if (!g_cond_wait_until (&self->mqtt_sink_gcond, &self->mqtt_sink_mutex,
            end_time)) {
      g_mutex_unlock (&self->mqtt_sink_mutex);
      g_critical ("Failed to connect to MQTT broker from mqttsink."
          "Please check broker is running status or broker host address.");
      goto error;
    }
  }
  g_mutex_unlock (&self->mqtt_sink_mutex);

  return TRUE;

error:
  MQTTAsync_destroy (&self->mqtt_client_handle);
  self->mqtt_client_handle = NULL;
  return FALSE;
}

/**
 * @brief Whether the state says the disconnect this element asked for is over
 */
static inline gboolean
_mqtt_sink_disconnected (mqtt_sink_state_t state)
{
  return ((state == MQTT_DISCONNECTED) || (state == MQTT_DISCONNECT_FAILED) ||
      (state == SINK_RENDER_EOS) || (state == SINK_RENDER_ERROR));
}

/**
 * @brief Stop mqttsink, called when state changed ready to null
 */
static gboolean
gst_mqtt_sink_stop (GstBaseSink * basesink)
{
  GstMqttSink *self = GST_MQTT_SINK (basesink);
  MQTTAsync_disconnectOptions disconn_opts =
      MQTTAsync_disconnectOptions_initializer;

  disconn_opts.timeout = DEFAULT_MQTT_DISCONNECT_TIMEOUT;
  disconn_opts.onSuccess = cb_mqtt_on_disconnect;
  disconn_opts.onFailure = cb_mqtt_on_disconnect_failure;
  disconn_opts.context = self;

  g_atomic_int_set (&self->mqtt_sink_state, SINK_RENDER_STOPPED);
  while (MQTTAsync_isConnected (self->mqtt_client_handle)) {
    gint64 end_time = g_get_monotonic_time () + DEFAULT_MQTT_DISCONNECT_TIMEOUT;
    mqtt_sink_state_t cur_state;

    MQTTAsync_disconnect (self->mqtt_client_handle, &disconn_opts);
    g_mutex_lock (&self->mqtt_sink_mutex);
    self->is_connected = FALSE;
    /**
     * The callback sets the state before it broadcasts, and it may have run
     * before this thread took the lock, so wait on the state rather than on
     * the signal alone.
     */
    while (!_mqtt_sink_disconnected (g_atomic_int_get (&self->mqtt_sink_state))) {
      if (!g_cond_wait_until (&self->mqtt_sink_gcond, &self->mqtt_sink_mutex,
              end_time))
        break;
    }
    g_mutex_unlock (&self->mqtt_sink_mutex);
    cur_state = g_atomic_int_get (&self->mqtt_sink_state);

    if (_mqtt_sink_disconnected (cur_state))
      break;
  }
  MQTTAsync_destroy (&self->mqtt_client_handle);
  self->mqtt_client_handle = NULL;
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
 * @brief A utility function to set the timestamp information onto the given buffer
 */
static void
_put_timestamp_to_msg_buf_hdr (GstMqttSink * self, GstBuffer * gst_buf,
    GstMQTTMessageHdr * hdr)
{
  hdr->base_time_epoch = self->base_time_epoch;
  hdr->sent_time_epoch = gst_mqtt_sink_get_epoch (self) *
      GST_US_TO_NS_MULTIPLIER;

  hdr->duration = GST_BUFFER_DURATION_IS_VALID (gst_buf) ?
      GST_BUFFER_DURATION (gst_buf) : GST_CLOCK_TIME_NONE;

  hdr->dts = GST_BUFFER_DTS_IS_VALID (gst_buf) ?
      GST_BUFFER_DTS (gst_buf) : GST_CLOCK_TIME_NONE;

  hdr->pts = GST_BUFFER_PTS_IS_VALID (gst_buf) ?
      GST_BUFFER_PTS (gst_buf) : GST_CLOCK_TIME_NONE;

  if (self->debug) {
    GstClockTime base_time = gst_element_get_base_time (GST_ELEMENT (self));
    gchar *topic = gst_mqtt_sink_dup_pub_topic (self);
    GstClock *clock;

    clock = gst_element_get_clock (GST_ELEMENT (self));

    GST_DEBUG_OBJECT (self,
        "%s now %" GST_TIME_FORMAT " ts %" GST_TIME_FORMAT " sent %"
        GST_TIME_FORMAT, topic,
        GST_TIME_ARGS (gst_clock_get_time (clock) - base_time),
        GST_TIME_ARGS (hdr->pts),
        GST_TIME_ARGS (hdr->sent_time_epoch - hdr->base_time_epoch));

    gst_object_unref (clock);
    g_free (topic);
  }
}

/**
 * @brief A utility function to set the message header
 */
static gboolean
_mqtt_set_msg_buf_hdr (GstBuffer * gst_buf, GstMQTTMessageHdr * hdr)
{
  gboolean ret = TRUE;
  guint i;

  hdr->num_mems = gst_buffer_n_memory (gst_buf);
  for (i = 0; i < hdr->num_mems; ++i) {
    GstMemory *each_mem;

    each_mem = gst_buffer_peek_memory (gst_buf, i);
    if (!each_mem) {
      memset (hdr, 0x0, sizeof (*hdr));
      ret = FALSE;
      break;
    }

    hdr->size_mems[i] = each_mem->size;
  }

  return ret;
}

/**
 * @brief The callback to process each buffer receiving on the sink pad
 */
static GstFlowReturn
gst_mqtt_sink_render (GstBaseSink * basesink, GstBuffer * in_buf)
{
  const gsize in_buf_size = gst_buffer_get_size (in_buf);
  GstMqttSink *self = GST_MQTT_SINK (basesink);
  GstFlowReturn ret = GST_FLOW_ERROR;
  mqtt_sink_state_t cur_state;
  GstMemory *in_buf_mem;
  GstMapInfo in_buf_map;
  gint mqtt_rc;
  guint8 *msg_pub;
  gchar *topic;
  gint64 end_time;

  /**
   * SINK_INITIALIZING is the only state that can still become
   * MQTT_CONNECTED: start () has waited for the connection, and nothing
   * reconnects once it is gone. So wait for that one state, holding the
   * lock the callbacks broadcast with while the state is read, and for at
   * most one 'pub-wait-timeout'. Every other state is the answer for this
   * buffer.
   */
  end_time = g_get_monotonic_time ();
  end_time += (self->mqtt_pub_wait_timeout * G_TIME_SPAN_SECOND);
  g_mutex_lock (&self->mqtt_sink_mutex);
  while (g_atomic_int_get (&self->mqtt_sink_state) == SINK_INITIALIZING) {
    if (!g_cond_wait_until (&self->mqtt_sink_gcond, &self->mqtt_sink_mutex,
            end_time))
      break;
  }
  g_mutex_unlock (&self->mqtt_sink_mutex);

  cur_state = g_atomic_int_get (&self->mqtt_sink_state);
  if (cur_state != MQTT_CONNECTED) {
    if (cur_state == SINK_RENDER_EOS) {
      ret = GST_FLOW_EOS;
    } else {
      g_printerr ("%s: Cannot publish a buffer: the connection to the broker "
          "is not up (state: %d)\n", TAG_ERR_MQTTSINK, cur_state);
      ret = GST_FLOW_ERROR;
    }
    goto ret_with;
  }

  if (self->num_buffers == 0) {
    ret = GST_FLOW_EOS;
    goto ret_with;
  }

  if (self->num_buffers != -1) {
    self->num_buffers -= 1;
  }

  if ((!self->is_static_sized_buf) && (self->mqtt_msg_buf) &&
      (self->mqtt_msg_buf_size != 0) &&
      (self->mqtt_msg_buf_size < in_buf_size + GST_MQTT_LEN_MSG_HDR)) {
    g_free (self->mqtt_msg_buf);
    self->mqtt_msg_buf = NULL;
    self->mqtt_msg_buf_size = 0;
  }

  /** Allocate a message buffer */
  if ((!self->mqtt_msg_buf) && (self->mqtt_msg_buf_size == 0)) {
    if (self->max_msg_buf_size == 0) {
      self->mqtt_msg_buf_size = in_buf_size + GST_MQTT_LEN_MSG_HDR;
    } else {
      self->mqtt_msg_buf_size = self->max_msg_buf_size + GST_MQTT_LEN_MSG_HDR;
      self->is_static_sized_buf = TRUE;
    }

    self->mqtt_msg_buf = g_try_malloc0 (self->mqtt_msg_buf_size);
  }

  if (!_mqtt_set_msg_buf_hdr (in_buf, &self->mqtt_msg_hdr)) {
    ret = GST_FLOW_ERROR;
    goto ret_with;
  }

  msg_pub = self->mqtt_msg_buf;
  if (!msg_pub) {
    self->mqtt_msg_buf_size = 0;
    ret = GST_FLOW_ERROR;
    goto ret_with;
  }

  if (self->mqtt_msg_buf_size < in_buf_size + GST_MQTT_LEN_MSG_HDR) {
    g_printerr ("%s: The given size for a message buffer is too small: "
        "given (%" G_GSIZE_FORMAT " bytes) vs. incoming (%" G_GSIZE_FORMAT
        " bytes)\n", TAG_ERR_MQTTSINK,
        self->mqtt_msg_buf_size - GST_MQTT_LEN_MSG_HDR, in_buf_size);
    ret = GST_FLOW_ERROR;
    goto ret_with;
  }
  memcpy (msg_pub, &self->mqtt_msg_hdr, sizeof (self->mqtt_msg_hdr));
  _put_timestamp_to_msg_buf_hdr (self, in_buf, (GstMQTTMessageHdr *) msg_pub);

  in_buf_mem = gst_buffer_get_all_memory (in_buf);
  if (!in_buf_mem) {
    ret = GST_FLOW_ERROR;
    goto ret_with;
  }

  if (!gst_memory_map (in_buf_mem, &in_buf_map, GST_MAP_READ)) {
    ret = GST_FLOW_ERROR;
    goto ret_unref_in_buf_mem;
  }

  ret = GST_FLOW_OK;

  memcpy (&msg_pub[sizeof (self->mqtt_msg_hdr)], in_buf_map.data,
      in_buf_map.size);
  topic = gst_mqtt_sink_dup_pub_topic (self);
  mqtt_rc = MQTTAsync_send (self->mqtt_client_handle, topic,
      GST_MQTT_LEN_MSG_HDR + in_buf_map.size, self->mqtt_msg_buf,
      self->mqtt_qos, 1, &self->mqtt_respn_opts);
  g_free (topic);
  if (mqtt_rc != MQTTASYNC_SUCCESS) {
    ret = GST_FLOW_ERROR;
  }

  gst_memory_unmap (in_buf_mem, &in_buf_map);

ret_unref_in_buf_mem:
  gst_memory_unref (in_buf_mem);

ret_with:
  return ret;
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
  GstMqttSink *self = GST_MQTT_SINK (basesink);
  GstEventType type = GST_EVENT_TYPE (event);
  gboolean ret = FALSE;

  switch (type) {
    case GST_EVENT_EOS:
      g_atomic_int_set (&self->mqtt_sink_state, SINK_RENDER_EOS);
      g_mutex_lock (&self->mqtt_sink_mutex);
      g_cond_broadcast (&self->mqtt_sink_gcond);
      g_mutex_unlock (&self->mqtt_sink_mutex);
      break;
    default:
      break;
  }

  ret = GST_BASE_SINK_CLASS (parent_class)->event (basesink, event);

  return ret;
}

/**
 * @brief An implementation of the set_caps vmethod in GstBaseSinkClass
 */
static gboolean
gst_mqtt_sink_set_caps (GstBaseSink * basesink, GstCaps * caps)
{
  GstMqttSink *self = GST_MQTT_SINK (basesink);
  gboolean ret;

  ret = gst_caps_replace (&self->in_caps, caps);

  if (ret && gst_caps_is_fixed (self->in_caps)) {
    gchar *caps_str = gst_caps_to_string (caps);
    gsize len;

    if (caps_str == NULL) {
      g_critical ("Fail to convert caps to string representation");
      return FALSE;
    }

    len = g_strlcpy (self->mqtt_msg_hdr.gst_caps_str, caps_str,
        GST_MQTT_MAX_LEN_GST_CAPS_STR);

    if (len >= GST_MQTT_MAX_LEN_GST_CAPS_STR) {
      g_critical ("Fail to copy caps_str.");
      ret = FALSE;
    }

    g_free (caps_str);
  }

  return ret;
}

/**
 * @brief Getter for the 'debug' property.
 */
static gboolean
gst_mqtt_sink_get_debug (GstMqttSink * self)
{
  return self->debug;
}

/**
 * @brief Setter for the 'debug' property.
 */
static void
gst_mqtt_sink_set_debug (GstMqttSink * self, const gboolean flag)
{
  self->debug = flag;
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
  gchar *new_id = g_strdup (id);
  gchar *old_id;

  GST_OBJECT_LOCK (self);
  old_id = self->mqtt_client_id;
  self->mqtt_client_id = new_id;
  GST_OBJECT_UNLOCK (self);

  g_free (old_id);
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
  gchar *new_addr = g_strdup (addr);
  gchar *old_addr;

  /**
   * @todo Handle the case where the addr is changed at runtime
   */
  GST_OBJECT_LOCK (self);
  old_addr = self->mqtt_host_address;
  self->mqtt_host_address = new_addr;
  GST_OBJECT_UNLOCK (self);

  g_free (old_addr);
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
  gchar *new_port = g_strdup (port);
  gchar *old_port;

  GST_OBJECT_LOCK (self);
  old_port = self->mqtt_host_port;
  self->mqtt_host_port = new_port;
  GST_OBJECT_UNLOCK (self);

  g_free (old_port);
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
  gchar *new_topic = g_strdup (topic);
  gchar *old_topic;

  GST_OBJECT_LOCK (self);
  old_topic = self->mqtt_topic;
  self->mqtt_topic = new_topic;
  GST_OBJECT_UNLOCK (self);

  g_free (old_topic);
}

/**
 * @brief Copy the 'pub-topic' property for a thread other than the one that sets it
 * @return a new string to be released with g_free ()
 */
static gchar *
gst_mqtt_sink_dup_pub_topic (GstMqttSink * self)
{
  gchar *topic;

  GST_OBJECT_LOCK (self);
  topic = g_strdup (self->mqtt_topic);
  GST_OBJECT_UNLOCK (self);

  return topic;
}

/**
 * @brief Getter for the 'cleansession' property.
 */
static gboolean
gst_mqtt_sink_get_opt_cleansession (GstMqttSink * self)
{
  return self->mqtt_conn_opts.cleansession;
}

/**
 * @brief Setter for the 'cleansession' property.
 */
static void
gst_mqtt_sink_set_opt_cleansession (GstMqttSink * self, const gboolean val)
{
  self->mqtt_conn_opts.cleansession = val;
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
  self->mqtt_pub_wait_timeout = to;
}

/**
 * @brief Getter for the 'keep-alive-interval' property
 */
static gint
gst_mqtt_sink_get_opt_keep_alive_interval (GstMqttSink * self)
{
  return self->mqtt_conn_opts.keepAliveInterval;
}

/**
 * @brief Setter for the 'keep-alive-interval' property
 */
static void
gst_mqtt_sink_set_opt_keep_alive_interval (GstMqttSink * self, const gint num)
{
  self->mqtt_conn_opts.keepAliveInterval = num;
}

/**
 * @brief Getter for the 'max-buffer-size' property.
 */
static gsize
gst_mqtt_sink_get_max_msg_buf_size (GstMqttSink * self)
{
  return self->max_msg_buf_size;
}

/**
 * @brief Setter for the 'max-buffer-size' property.
 */
static void
gst_mqtt_sink_set_max_msg_buf_size (GstMqttSink * self, const gsize size)
{
  self->max_msg_buf_size = size;
}

/**
 * @brief Getter for the 'num-buffers' property.
 */
static gint
gst_mqtt_sink_get_num_buffers (GstMqttSink * self)
{
  gint num_buffers;

  num_buffers = self->num_buffers;

  return num_buffers;
}

/**
 * @brief Setter for the 'num-buffers' property
 */
static void
gst_mqtt_sink_set_num_buffers (GstMqttSink * self, const gint num)
{
  self->num_buffers = num;
}

/**
 * @brief Getter for the 'mqtt-qos' property.
 */
static gint
gst_mqtt_sink_get_mqtt_qos (GstMqttSink * self)
{
  return self->mqtt_qos;
}

/**
 * @brief Setter for the 'mqtt-qos' property
 */
static void
gst_mqtt_sink_set_mqtt_qos (GstMqttSink * self, const gint qos)
{
  self->mqtt_qos = qos;
}

/**
 * @brief Getter for the 'ntp-sync' property.
 */
static gboolean
gst_mqtt_sink_get_mqtt_ntp_sync (GstMqttSink * self)
{
  return self->mqtt_ntp_sync;
}

/**
 * @brief Setter for the 'ntp-sync' property
 */
static void
gst_mqtt_sink_set_mqtt_ntp_sync (GstMqttSink * self, const gboolean flag)
{
  self->mqtt_ntp_sync = flag;
}

/**
 * @brief Getter for the 'ntp-srvs' property.
 */
static gchar *
gst_mqtt_sink_get_mqtt_ntp_srvs (GstMqttSink * self)
{
  return self->mqtt_ntp_srvs;
}

/**
 * @brief Setter for the 'ntp-srvs' property
 */
static void
gst_mqtt_sink_set_mqtt_ntp_srvs (GstMqttSink * self, const gchar * pairs)
{
  GstMqttNtpServers *srvs;
  GstMqttNtpServers *old_srvs;
  gchar *old_pairs;

  srvs = gst_mqtt_ntp_servers_new (pairs);
  if (!srvs)
    return;

  GST_OBJECT_LOCK (self);
  old_pairs = self->mqtt_ntp_srvs;
  old_srvs = self->mqtt_ntp_servers;
  self->mqtt_ntp_srvs = g_strdup (pairs);
  self->mqtt_ntp_servers = srvs;
  GST_OBJECT_UNLOCK (self);

  g_free (old_pairs);
  gst_mqtt_ntp_servers_unref (old_srvs);
}

/**
 * @brief Parse HOST_NAME:PORT pairs into a new NTP server list, dropping a
 *        pair that has no valid port.
 * @return the list holding one reference, or NULL if memory runs out
 */
static GstMqttNtpServers *
gst_mqtt_ntp_servers_new (const gchar * pairs)
{
  GstMqttNtpServers *srvs;
  gchar **pair_arrs;
  guint hnum;
  guint i;

  srvs = g_try_new0 (GstMqttNtpServers, 1);
  if (!srvs)
    return NULL;

  srvs->ref_count = 1;
  pair_arrs = g_strsplit (pairs ? pairs : "", ",", -1);
  hnum = g_strv_length (pair_arrs);
  srvs->hnames = g_try_new0 (gchar *, hnum + 1);
  srvs->ports = g_try_new0 (guint16, hnum + 1);
  if (!srvs->hnames || !srvs->ports) {
    g_strfreev (pair_arrs);
    gst_mqtt_ntp_servers_unref (srvs);
    return NULL;
  }

  for (i = 0; i < hnum; i++) {
    gchar **hname_port = g_strsplit (pair_arrs[i], ":", 2);
    gulong port_ul = 0;

    if (g_strv_length (hname_port) == 2)
      port_ul = strtoul (hname_port[1], NULL, 10);
    if ((port_ul != 0) && (port_ul <= UINT16_MAX)) {
      srvs->hnames[srvs->num] = g_strdup (hname_port[0]);
      srvs->ports[srvs->num] = (guint16) port_ul;
      srvs->num++;
    }

    g_strfreev (hname_port);
  }

  g_strfreev (pair_arrs);
  return srvs;
}

/**
 * @brief Drop a reference to an NTP server list, freeing it with the last one
 */
static void
gst_mqtt_ntp_servers_unref (GstMqttNtpServers * srvs)
{
  if (!srvs || !g_atomic_int_dec_and_test (&srvs->ref_count))
    return;

  g_strfreev (srvs->hnames);
  g_free (srvs->ports);
  g_free (srvs);
}

/**
 * @brief Get the Unix epoch from get_epoch_func () while holding a reference
 *        to the NTP server list it reads
 */
static int64_t
gst_mqtt_sink_get_epoch (GstMqttSink * self)
{
  GstMqttNtpServers *srvs;
  int64_t epoch;

  GST_OBJECT_LOCK (self);
  srvs = self->mqtt_ntp_servers;
  if (srvs)
    g_atomic_int_inc (&srvs->ref_count);
  GST_OBJECT_UNLOCK (self);

  if (!srvs)
    return self->get_epoch_func (0, NULL, NULL);

  epoch = self->get_epoch_func (srvs->num, srvs->hnames, srvs->ports);
  gst_mqtt_ntp_servers_unref (srvs);

  return epoch;
}

/** Callback function definitions */
/**
 * @brief A callback function corresponding to MQTTAsync_connectOptions's
 *        onSuccess. This callback is invoked when the connection between
 *        this element and the broker is properly established.
 */
static void
cb_mqtt_on_connect (void *context, MQTTAsync_successData * response)
{
  GstMqttSink *self = (GstMqttSink *) context;
  UNUSED (response);

  g_atomic_int_set (&self->mqtt_sink_state, MQTT_CONNECTED);
  g_mutex_lock (&self->mqtt_sink_mutex);
  self->is_connected = TRUE;
  g_cond_broadcast (&self->mqtt_sink_gcond);
  g_mutex_unlock (&self->mqtt_sink_mutex);
}

/**
 * @brief A callback function corresponding to MQTTAsync_connectOptions's
 *        onFailure. This callback is invoked when it is failed to connect to
 *        the broker.
 */
static void
cb_mqtt_on_connect_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSink *self = (GstMqttSink *) context;
  UNUSED (response);

  g_atomic_int_set (&self->mqtt_sink_state, MQTT_CONNECT_FAILURE);
  g_mutex_lock (&self->mqtt_sink_mutex);
  self->is_connected = FALSE;
  g_cond_broadcast (&self->mqtt_sink_gcond);
  g_mutex_unlock (&self->mqtt_sink_mutex);
}

/**
 * @brief A callback function corresponding to MQTTAsync_disconnectOptions's
 *        onSuccess. Regardless of the MQTTAsync_disconnect function's result,
 *        the pipeline should be stopped after this callback.
 */
static void
cb_mqtt_on_disconnect (void *context, MQTTAsync_successData * response)
{
  GstMqttSink *self = (GstMqttSink *) context;
  UNUSED (response);

  g_atomic_int_set (&self->mqtt_sink_state, MQTT_DISCONNECTED);
  g_mutex_lock (&self->mqtt_sink_mutex);
  g_cond_broadcast (&self->mqtt_sink_gcond);
  g_mutex_unlock (&self->mqtt_sink_mutex);
}

/**
 * @brief A callback function corresponding to MQTTAsync_disconnectOptions's
 *        onFailure. Regardless of the MQTTAsync_disconnect function's result,
 *        the pipeline should be stopped after this callback.
 */
static void
cb_mqtt_on_disconnect_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSink *self = (GstMqttSink *) context;
  UNUSED (response);

  g_atomic_int_set (&self->mqtt_sink_state, MQTT_DISCONNECT_FAILED);
  g_mutex_lock (&self->mqtt_sink_mutex);
  g_cond_broadcast (&self->mqtt_sink_gcond);
  g_mutex_unlock (&self->mqtt_sink_mutex);
}

/**
 * @brief A callback function to be given to the MQTTAsync_setCallbacks function.
 *        This callback is activated when `mqtt-qos` is higher then 0.
 */
static void
cb_mqtt_on_delivery_complete (void *context, MQTTAsync_token token)
{
  GstMqttSink *self = (GstMqttSink *) context;
  gchar *topic = gst_mqtt_sink_dup_pub_topic (self);

  GST_DEBUG_OBJECT (self,
      "%s: the message with token(%d) has been delivered.", topic, token);
  g_free (topic);
}

/**
 * @brief A callback function to be given to the MQTTAsync_setCallbacks function.
 *        When the connection between this element and the broker is broken,
 *        this callback will be invoked.
 */
static void
cb_mqtt_on_connection_lost (void *context, char *cause)
{
  GstMqttSink *self = (GstMqttSink *) context;
  UNUSED (cause);

  g_atomic_int_set (&self->mqtt_sink_state, MQTT_CONNECTION_LOST);
  g_mutex_lock (&self->mqtt_sink_mutex);
  self->is_connected = FALSE;
  g_cond_broadcast (&self->mqtt_sink_gcond);
  g_mutex_unlock (&self->mqtt_sink_mutex);
}

/**
 * @brief A callback function to be given to the MQTTAsync_setCallbacks function.
 *        In the case of the publisher, this callback is not used.
 */
static int
cb_mqtt_on_message_arrived (void *context, char *topicName, int topicLen,
    MQTTAsync_message * message)
{
  UNUSED (context);
  UNUSED (topicName);
  UNUSED (topicLen);
  UNUSED (message);
  /* nothing to do */
  return 1;
}

/**
 * @brief A callback function corresponding to MQTTAsync_responseOptions's
 *        onSuccess.
 */
static void
cb_mqtt_on_send_success (void *context, MQTTAsync_successData * response)
{
  GstMqttSink *self = (GstMqttSink *) context;
  mqtt_sink_state_t state = g_atomic_int_get (&self->mqtt_sink_state);
  UNUSED (response);

  if (state == SINK_RENDER_STOPPED) {
    g_atomic_int_set (&self->mqtt_sink_state, SINK_RENDER_EOS);

    g_mutex_lock (&self->mqtt_sink_mutex);
    g_cond_broadcast (&self->mqtt_sink_gcond);
    g_mutex_unlock (&self->mqtt_sink_mutex);
  }
}

/**
 * @brief A callback function corresponding to MQTTAsync_responseOptions's
 *        onFailure.
 */
static void
cb_mqtt_on_send_failure (void *context, MQTTAsync_failureData * response)
{
  GstMqttSink *self = (GstMqttSink *) context;
  mqtt_sink_state_t state = g_atomic_int_get (&self->mqtt_sink_state);
  UNUSED (response);

  if (state == SINK_RENDER_STOPPED) {
    g_atomic_int_set (&self->mqtt_sink_state, SINK_RENDER_ERROR);

    g_mutex_lock (&self->mqtt_sink_mutex);
    g_cond_broadcast (&self->mqtt_sink_gcond);
    g_mutex_unlock (&self->mqtt_sink_mutex);
  }

}
