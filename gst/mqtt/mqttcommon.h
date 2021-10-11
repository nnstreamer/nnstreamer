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
#include <stdint.h>

#ifndef UNUSED
#define UNUSED(expr) do { (void)(expr); } while (0)
#endif /* UNUSED */

#ifndef GST_MQTT_PACKAGE
#define GST_MQTT_PACKAGE "GStreamer MQTT Plugins"
#endif /* GST_MQTT_PACKAGE */

#define GST_MQTT_ELEM_NAME_SINK "mqttsink"
#define GST_MQTT_ELEM_NAME_SRC "mqttsrc"

#define GST_MQTT_LEN_MSG_HDR          1024
#define GST_MQTT_MAX_LEN_GST_CAPS_STR 512
/**
 * @brief GST_BUFFER_MEM_MAX in gstreamer/gstbuffer.c is 16. To represent each
 *        size of the memory block that the GstBuffer contains, GST_MQTT_MAX_NUM_MEMS
 *        should be 16.
 */
#define GST_MQTT_MAX_NUM_MEMS   16

#define GST_US_TO_NS_MULTIPLIER 1000

#define DEFAULT_MQTT_CONN_TIMEOUT_SEC 5

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
      gchar gst_caps_str[GST_MQTT_MAX_LEN_GST_CAPS_STR];
    };
    guint8 _reserved_hdr[GST_MQTT_LEN_MSG_HDR];
  };
} GstMQTTMessageHdr;

typedef int64_t (*mqtt_get_unix_epoch)(uint32_t, char **, uint16_t *);

/**
 * @brief A wrapper function of g_get_real_time () to assign it to the function
 * pointer, mqtt_get_unix_epoch
 */
static inline int64_t default_mqtt_get_unix_epoch (uint32_t hnum, char **hnames,
    uint16_t *hports)
{
  UNUSED (hnum);
  UNUSED (hnames);
  UNUSED (hports);
  return g_get_real_time ();
}

#endif /* !__GST_MQTT_COMMON_H__ */
