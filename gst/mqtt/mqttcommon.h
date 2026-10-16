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
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#include <glib.h>
#include <MQTTAsync.h>

/** GRND_NONBLOCK of getrandom (), which older C libraries do not define */
#define GST_MQTT_GRND_NONBLOCK 0x0001

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
 * @brief The generator of the default client id, gst_mqtt_new_client_id () unless replaced by a test
 * @note start () calls it with the object lock held, so it must not access the element.
 */
typedef gchar *(*mqtt_new_client_id) (const gchar * prefix);

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
 * @brief Fill a buffer from the secure random source of the operating system.
 * @param buf The buffer to fill
 * @param len The number of bytes to fill
 * @param device The random device to read if getrandom () is not used or not available
 * @param try_getrandom Whether to try the getrandom () system call (Linux) first
 * @return TRUE if the whole buffer is filled, FALSE if no source could fill it.
 *
 * getrandom () needs no file descriptor, so it also works in a chroot without
 * the device and when the process has run out of file descriptors. Unlike
 * g_rand_new (), it never falls back to a seed made of the time and the PIDs.
 * It does not block: before the kernel pool is ready (early boot) it fails
 * and the device is read instead, as before.
 */
static inline gboolean
gst_mqtt_read_random (guint8 * buf, gsize len, const gchar * device,
    gboolean try_getrandom)
{
  gsize filled = 0;
  int fd;

#ifdef SYS_getrandom
  while (try_getrandom && filled < len) {
    long ret = syscall (SYS_getrandom, buf + filled, len - filled,
        GST_MQTT_GRND_NONBLOCK);

    if (ret > 0)
      filled += (gsize) ret;
    else if (ret < 0 && errno == EINTR)
      continue;
    else
      break;
  }
  if (filled == len)
    return TRUE;
  filled = 0;
#else
  UNUSED (try_getrandom);
#endif

  fd = open (device, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return FALSE;

  while (filled < len) {
    ssize_t ret = read (fd, buf + filled, len - filled);

    if (ret > 0)
      filled += (gsize) ret;
    else if (ret < 0 && errno == EINTR)
      continue;
    else
      break;
  }
  close (fd);

  return (filled == len);
}

/**
 * @brief Generate a client id from the given random source.
 * @param prefix The prefix naming the element type, of [0-9a-z] only
 * @param device The random device, see gst_mqtt_read_random ()
 * @param try_getrandom Whether to try getrandom () first, see gst_mqtt_read_random ()
 * @return A newly allocated id, the prefix followed by 64 random bits in hex,
 *         or NULL if the random source is not available.
 *         The caller frees it with g_free ().
 */
static inline gchar *
gst_mqtt_new_client_id_from (const gchar * prefix, const gchar * device,
    gboolean try_getrandom)
{
  guint8 bytes[8];
  guint64 bits = 0;
  guint i;

  if (!gst_mqtt_read_random (bytes, sizeof (bytes), device, try_getrandom))
    return NULL;

  for (i = 0; i < sizeof (bytes); i++)
    bits = (bits << 8) | bytes[i];

  return g_strdup_printf ("%s%016" G_GINT64_MODIFIER "x", prefix, bits);
}

/**
 * @brief Generate a client id for an element whose client-id is not set.
 * @param prefix The prefix naming the element type, of [0-9a-z] only
 * @return A newly allocated id, the prefix followed by 64 random bits in hex,
 *         or NULL if the secure random source of the system is not available.
 *         The caller frees it with g_free ().
 *
 * The id is not derived from the host name or the process id, so a remote
 * party cannot guess it and take over the session with the broker. The bits
 * come from the operating system (gst_mqtt_read_random ()), not from a GLib
 * generator, which an application seed or a missing /dev/urandom would make
 * predictable.
 * With a prefix of up to 7 characters, the id stays within the 23 characters
 * of [0-9a-zA-Z] that every MQTT 3.1.1 broker must accept.
 */
static inline gchar *
gst_mqtt_new_client_id (const gchar * prefix)
{
  return gst_mqtt_new_client_id_from (prefix, "/dev/urandom", TRUE);
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
