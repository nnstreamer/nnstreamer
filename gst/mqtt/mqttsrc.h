/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsrc.h
 * @date    08 Mar 2021
 * @brief   Subscribe a MQTT topic and push incoming data to the GStreamer pipeline
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_MQTT_SRC_H__
#define __GST_MQTT_SRC_H__
#include <gst/base/gstbasesrc.h>
#include <gst/base/gstdataqueue.h>
#include <gst/gst.h>
#include <MQTTAsync.h>

#include "mqttcommon.h"

G_BEGIN_DECLS

#define GST_TYPE_MQTT_SRC \
    (gst_mqtt_src_get_type())
#define GST_MQTT_SRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_MQTT_SRC, GstMqttSrc))
#define GST_IS_MQTT_SRC(obj)  \
    (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_MQTT_SRC))
#define GST_MQTT_SRC_CAST(obj) \
    ((GstMqttSrc *) obj)
#define GST_MQTT_SRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_MQTT_SRC, GstMqttSrcClass))
#define GST_IS_MQTT_SRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_MQTT_SRC))

typedef struct _GstMqttSrc GstMqttSrc;
typedef struct _GstMqttSrcClass GstMqttSrcClass;

/**
 * @brief GstMqttSrc data structure.
 *
 * GstMqttSrc inherits GstBaseSrc.
 */
struct _GstMqttSrc {
  GstBaseSrc parent;
  GstCaps *caps;
  GQuark gquark_err_tag;
  GError *err;
  gint64 base_time_epoch;
  GstClockTime latency;
  gchar *mqtt_client_id;
  gchar *mqtt_host_address;
  gchar *mqtt_host_port;
  gchar *mqtt_topic;
  gint64 mqtt_sub_timeout;
  gboolean debug;
  gboolean is_live;
  guint64 num_dumped;
  gint mqtt_qos;

  GAsyncQueue *aqueue;
  GMutex mqtt_src_mutex;
  GCond mqtt_src_gcond;
  gboolean is_connected;
  gboolean is_subscribed;

  MQTTAsync mqtt_client_handle;
  MQTTAsync_connectOptions mqtt_conn_opts;
  MQTTAsync_responseOptions mqtt_respn_opts;
};

/**
 * @brief GstMqttSrcClass data structure.
 *
 * GstMqttSrcClass inherits GstBaseSrcClass.
 */
struct _GstMqttSrcClass {
  GstBaseSrcClass parent_class;
};

GType gst_mqtt_src_get_type (void);

G_END_DECLS
#endif /* !__GST_MQTT_SRC_H__ */
