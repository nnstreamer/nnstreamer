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
#include <MQTTClient.h>

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
 * @brief GstMqttSink data structure.
 *
 * GstMqttSink inherits GstBaseSink.
 */
struct _GstMqttSink {
  GstBaseSink parent;
  guint num_buffers;
  GQuark gquark_err_tag;
  GError *err;
  MQTTClient *mqtt_client_handle;
  MQTTClient_connectOptions *mqtt_conn_opts;
  gchar *mqtt_client_id;
  gchar *mqtt_host_address;
  gchar *mqtt_host_port;
  gchar *mqtt_topic;
  gulong mqtt_pub_wait_timeout;
  gboolean mqtt_msg_hdr_update_flag;
  GstMQTTMessageHdr *mqtt_msg_hdr;
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
