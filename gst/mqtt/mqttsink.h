/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsink.h
 * @date    08 Mar 2021
 * @brief   Publish incoming data streams as a MQTT topic
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_MQTT_SINK_H__
#define __GST_MQTT_SINK_H__
#include <gst/base/gstbasesink.h>
#include <gst/gst.h>
#include <MQTTAsync.h>

#include "mqttcommon.h"

G_BEGIN_DECLS

#define GST_TYPE_MQTT_SINK \
    (gst_mqtt_sink_get_type())
#define GST_MQTT_SINK(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_MQTT_SINK, GstMqttSink))
#define GST_IS_MQTT_SINK(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_MQTT_SINK))
#define GST_MQTT_SINK_CAST(obj) \
    ((GstMqttSink *) obj)
#define GST_MQTT_SINK_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_MQTT_SINK, GstMqttSinkClass))
#define GST_IS_MQTT_SINK_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_MQTT_SINK))

typedef struct _GstMqttSink GstMqttSink;
typedef struct _GstMqttSinkClass GstMqttSinkClass;

/**
 * @brief A type definition to indicate the state of this element
 */
typedef enum _mqtt_sink_state_t {
  MQTT_CONNECTION_LOST = -3,
  MQTT_CONNECT_FAILURE = -2,
  SINK_INITIALIZING = -1,
  SINK_RENDER_STOPPED = 0,
  SINK_RENDER_EOS,
  SINK_RENDER_ERROR,
  MQTT_CONNECTED,
  MQTT_DISCONNECTED,
  MQTT_DISCONNECT_FAILED,
} mqtt_sink_state_t;

/**
 * @brief GstMqttSink data structure.
 *
 * GstMqttSink inherits GstBaseSink.
 */
struct _GstMqttSink {
  GstBaseSink parent;
  GstCaps *in_caps;
  gint num_buffers;
  gsize max_msg_buf_size;
  GQuark gquark_err_tag;
  GError *err;
  gint64 base_time_epoch;
  gchar *mqtt_client_id;
  gchar *mqtt_host_address;
  gchar *mqtt_host_port;
  gchar *mqtt_topic;
  gulong mqtt_pub_wait_timeout;
  GMutex mqtt_sink_mutex;
  GCond mqtt_sink_gcond;
  mqtt_sink_state_t mqtt_sink_state;
  gboolean debug;
  gint mqtt_qos;
  gboolean mqtt_ntp_sync;
  guint mqtt_ntp_num_srvs;
  gchar *mqtt_ntp_srvs;
  gchar **mqtt_ntp_hnames;
  guint16 *mqtt_ntp_ports;
  gboolean is_connected;

  mqtt_get_unix_epoch get_epoch_func;

  GstMQTTMessageHdr mqtt_msg_hdr;
  gpointer mqtt_msg_buf;
  gsize mqtt_msg_buf_size;

  MQTTAsync mqtt_client_handle;
  MQTTAsync_connectOptions mqtt_conn_opts;
  MQTTAsync_responseOptions mqtt_respn_opts;
};

/**
 * @brief GstMqttSinkClass data structure.
 *
 * GstMqttSinkClass inherits GstBaseSinkClass.
 */
struct _GstMqttSinkClass {
  GstBaseSinkClass parent_class;
};

GType gst_mqtt_sink_get_type (void);

G_END_DECLS
#endif /* !__GST_MQTT_SINK_H__ */
