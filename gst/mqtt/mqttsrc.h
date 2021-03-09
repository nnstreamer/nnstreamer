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
#include <gst/gst.h>

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

struct _GstMqttSrc {
  GstBaseSrc parent;
};

struct _GstMqttSrcClass {
  GstBaseSrcClass parent_class;
};

GType gst_mqtt_src_get_type (void);

G_END_DECLS
#endif /* !__GST_MQTT_SRC_H__ */
