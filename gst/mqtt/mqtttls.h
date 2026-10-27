/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file mqtttls.h
 * @brief Shared TLS properties and connection snapshots for MQTT elements
 * @author Minsoo Kim
 * @bug No known bugs
 */

#ifndef __GST_MQTT_TLS_H__
#define __GST_MQTT_TLS_H__

#include <glib-object.h>
#include <string.h>
#include <MQTTAsync.h>

#define GST_MQTT_TLS_NUM_PROPERTIES 5

/** @brief Configured properties and the separately owned reconnect snapshot. */
typedef struct
{
  gboolean enabled;
  gchar *files[4];
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

/** @brief Install the same opt-in TLS properties on either element. */
static inline void
gst_mqtt_tls_install_properties (GObjectClass *klass, guint first)
{
  const gchar *names[] = { "tls-ca-file", "tls-cert-file", "tls-key-file",
    "tls-key-password"
  };
  const gchar *descriptions[] = {
    "PEM CA trust store; NULL uses the TLS library default trust store",
    "PEM client certificate chain, optionally including its private key",
    "PEM client private key; requires tls-cert-file",
    "Password for the client private key; cannot be read back"
  };
  guint i;

  g_object_class_install_property (klass, first,
      g_param_spec_boolean ("tls", "TLS",
          "Use TLS and verify the broker certificate and hostname", FALSE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  for (i = 0; i < G_N_ELEMENTS (names); i++) {
    GParamFlags flags = i == 3 ? G_PARAM_WRITABLE : G_PARAM_READWRITE;
    g_object_class_install_property (klass, first + i + 1,
        g_param_spec_string (names[i], names[i], descriptions[i], NULL,
            (GParamFlags) (flags | G_PARAM_STATIC_STRINGS)));
  }
}

/** @brief Set a configured value; the caller holds GST_OBJECT_LOCK. */
static inline void
gst_mqtt_tls_set_property (GstMqttTls *tls, guint index, const GValue *value)
{
  if (index == 0) {
    tls->enabled = g_value_get_boolean (value);
  } else {
    g_free (tls->files[index - 1]);
    tls->files[index - 1] = g_value_dup_string (value);
  }
}

/** @brief Read a non-secret configured value with GST_OBJECT_LOCK held. */
static inline void
gst_mqtt_tls_get_property (GstMqttTls *tls, guint index, GValue *value)
{
  if (index == 0)
    g_value_set_boolean (value, tls->enabled);
  else if (index < 4)
    g_value_set_string (value, tls->files[index - 1]);
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

  if (tls->files[2] && !tls->files[1])
    return "A TLS private key requires tls-cert-file.";
  for (i = 0; i < 3; i++) {
    if (tls->files[i] && !g_file_test (tls->files[i], G_FILE_TEST_IS_REGULAR))
      return "A configured TLS certificate or key file is not a regular file.";
  }
  tls->options.trustStore = g_strdup (tls->files[0]);
  tls->options.keyStore = g_strdup (tls->files[1]);
  tls->options.privateKey = g_strdup (tls->files[2]);
  tls->options.privateKeyPassword = g_strdup (tls->files[3]);
  opts->ssl = &tls->options;
  return NULL;
}

/** @brief Free configured values and the inactive connection snapshot. */
static inline void
gst_mqtt_tls_clear (GstMqttTls *tls)
{
  guint i;
  gst_mqtt_tls_clear_snapshot (tls);
  for (i = 0; i < G_N_ELEMENTS (tls->files); i++)
    g_clear_pointer (&tls->files[i], g_free);
}

#endif /* __GST_MQTT_TLS_H__ */
