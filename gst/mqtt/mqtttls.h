/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file mqtttls.h
 * @brief Shared TLS settings and connection snapshots for MQTT elements
 * @author Minsoo Kim
 * @bug No known bugs
 */

#ifndef __GST_MQTT_TLS_H__
#define __GST_MQTT_TLS_H__

#include <glib-object.h>
#include <string.h>
#include <MQTTAsync.h>

/** @brief Configured properties and the separately owned reconnect snapshot. */
typedef struct
{
  gboolean enabled;
  gchar *ca_file;
  gchar *cert_file;
  gchar *key_file;
  gchar *key_password;
  MQTTAsync_SSLOptions options;
} GstMqttTls;

/** @brief Initialize the TLS options without enabling TLS. */
static inline void
gst_mqtt_tls_init (GstMqttTls *tls)
{
  MQTTAsync_SSLOptions options = MQTTAsync_SSLOptions_initializer;
  tls->options = options;
  tls->options.enableServerCertAuth = 1;
  tls->options.verify = 1;
}

/** @brief Release the options only after the previous client is destroyed. */
static inline void
gst_mqtt_tls_clear_snapshot (GstMqttTls *tls)
{
  g_clear_pointer ((gchar **) & tls->options.trustStore, g_free);
  g_clear_pointer ((gchar **) & tls->options.keyStore, g_free);
  g_clear_pointer ((gchar **) & tls->options.privateKey, g_free);
  g_clear_pointer ((gchar **) & tls->options.privateKeyPassword, g_free);
}

/**
 * @brief Validate and snapshot TLS settings before creating a new client.
 * @return NULL on success, or a static diagnostic on invalid settings.
 * @note The caller holds GST_OBJECT_LOCK; property updates never free this snapshot.
 */
static inline const gchar *
gst_mqtt_tls_prepare (GstMqttTls *tls, MQTTAsync_connectOptions *opts,
    const gchar *host)
{
  const gchar *files[] = { tls->ca_file, tls->cert_file, tls->key_file };
  guint i;

  if (!host || !*host)
    return "The broker host must not be empty.";
  if (strstr (host, "://") && (tls->enabled ||
          (!g_str_has_prefix (host, "tcp://") &&
              !g_str_has_prefix (host, "mqtt://") &&
              !g_str_has_prefix (host, "ws://"))))
    return "Use a bare broker hostname and tls=true for verified TLS.";

  opts->ssl = NULL;
  gst_mqtt_tls_clear_snapshot (tls);
  if (!tls->enabled)
    return NULL;

#ifndef ENABLE_MQTT_TLS
  return "TLS is not built in; configure with -Dmqtt-tls=true.";
#endif

  if (tls->key_file && !tls->cert_file)
    return "A TLS private key requires tls-cert-file.";
  for (i = 0; i < G_N_ELEMENTS (files); i++) {
    if (files[i] && !g_file_test (files[i], G_FILE_TEST_IS_REGULAR))
      return "A configured TLS certificate or key file is not a regular file.";
  }
  tls->options.trustStore = g_strdup (tls->ca_file);
  tls->options.keyStore = g_strdup (tls->cert_file);
  tls->options.privateKey = g_strdup (tls->key_file);
  tls->options.privateKeyPassword = g_strdup (tls->key_password);
  opts->ssl = &tls->options;
  return NULL;
}

/** @brief Free configured values and the inactive connection snapshot. */
static inline void
gst_mqtt_tls_clear (GstMqttTls *tls)
{
  gst_mqtt_tls_clear_snapshot (tls);
  g_clear_pointer (&tls->ca_file, g_free);
  g_clear_pointer (&tls->cert_file, g_free);
  g_clear_pointer (&tls->key_file, g_free);
  g_clear_pointer (&tls->key_password, g_free);
}

#endif /* __GST_MQTT_TLS_H__ */
