/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttcommon.h
 * @date    08 Mar 2021
 * @brief   Common macros and utility functions for GStreamer MQTT plugins
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __GST_MQTT_COMMON_H__
#define __GST_MQTT_COMMON_H__

#ifndef GST_MQTT_PACKAGE
#define GST_MQTT_PACKAGE "GStreamer MQTT Plugins"
#endif /* GST_MQTT_PACKAGE */

#define GST_MQTT_ELEM_NAME_SINK "mqttsink"
#define GST_MQTT_ELEM_NAME_SRC "mqttsrc"

#define GST_MQTT_LEN_MSG_HDR    512

/**
 * @brief GST_BUFFER_MEM_MAX in gstreamer/gstbuffer.c is 16. To represent each
 *        size of the memory block that the GstBuffer contains, GST_MQTT_MAX_NUM_MEMS
 *        should be 16.
 */
#define GST_MQTT_MAX_NUM_MEMS   16

#define GST_US_TO_NS_MULTIPLIER 1000

/**
 * @brief Defined a custom data type, GstMQTTMessageHdr
 *
 * GstMQTTMessageHdr contains the information needed to parse the message data
 * at the subscriber side and is prepended to the original message data at the
 * publisher side.
 */
typedef struct _GstMQTTMessageHdr {
  union {
    struct {
      guint num_mems;
      gsize size_mems[GST_MQTT_MAX_NUM_MEMS];
      gint64 base_time_epoch;
      gint64 sent_time_epoch;
      GstClockTime duration;
      GstClockTime dts;
      GstClockTime pts;
    };
    guint8 _reserved_hdr[GST_MQTT_LEN_MSG_HDR];
  };
} GstMQTTMessageHdr;

#endif /* !__GST_MQTT_COMMON_H__ */
