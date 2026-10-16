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
#include <glib.h>
#include <MQTTAsync.h>

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

/**
 * @brief Generate a client id for an element whose client-id is not set.
 * @param prefix The prefix naming the element type, of [0-9a-z] only
 * @return A newly allocated id, the prefix followed by 64 random bits in hex.
 *         The caller frees it with g_free ().
 *
 * The id is not derived from the host name or the process id, so a remote
 * party cannot guess it and take over the session with the broker. The random
 * source is a new GRand, seeded from the system entropy, so that a fixed seed
 * given to g_random_set_seed () by the application cannot make it predictable.
 * Where GLib cannot read the system entropy (no /dev/urandom), it seeds the
 * GRand from the time and the process id instead.
 * With a prefix of up to 7 characters, the id stays within the 23 characters
 * of [0-9a-zA-Z] that every MQTT 3.1.1 broker must accept.
 */
static inline gchar *
gst_mqtt_new_client_id (const gchar * prefix)
{
  GRand *rand = g_rand_new ();
  guint32 high = g_rand_int (rand);
  guint32 low = g_rand_int (rand);

  g_rand_free (rand);
  return g_strdup_printf ("%s%08x%08x", prefix, high, low);
}

/**
 * @brief Replace the credentials of the connect options with copies of the given ones.
 * @param opts The connect options that own the copies
 * @param username The user name, or NULL to connect without one
 * @param password The password, or NULL to connect without one
 *
 * The copies stay valid until the next call, as the client may use them again
 * to reconnect. Call it with NULL for both to free them.
 */
static inline void
gst_mqtt_replace_credentials (MQTTAsync_connectOptions * opts,
    const gchar * username, const gchar * password)
{
  g_free ((gchar *) opts->username);
  g_free ((gchar *) opts->password);
  opts->username = g_strdup (username);
  opts->password = g_strdup (password);
}

#endif /* !__GST_MQTT_COMMON_H__ */
