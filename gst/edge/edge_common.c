/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_common.c
 * @date    01 Aug 2022
 * @brief   Common functions for edge sink and src
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "edge_common.h"
#include "nnstreamer_log.h"

/**
 * @brief register GEnumValue array for edge protocol property handling
 */
GType
gst_edge_get_connect_type (void)
{
  static GType protocol = 0;
  if (protocol == 0) {
    static GEnumValue protocols[] = {
      {NNS_EDGE_CONNECT_TYPE_TCP, "TCP",
          "Directly sending stream frames via TCP connections."},
      {NNS_EDGE_CONNECT_TYPE_HYBRID, "HYBRID",
          "Connect with MQTT brokers and directly sending stream frames via TCP connections."},
      {NNS_EDGE_CONNECT_TYPE_MQTT, "MQTT",
          "Sending stream frames via MQTT connections."},
      {NNS_EDGE_CONNECT_TYPE_CUSTOM, "CUSTOM",
          "Sending stream frames via CUSTOM connections."},
      {0, NULL, NULL},
    };
    protocol = g_enum_register_static ("edge_protocol", protocols);
  }

  return protocol;
}

/**
 * @brief Parse custom properties and set them on the edge handle.
 */
gboolean
gst_edge_parse_custom_props (nns_edge_h edge_h, const gchar * custom_props)
{
  gchar **tokens;
  guint i;
  gboolean all_set = TRUE;

  g_return_val_if_fail (edge_h != NULL, FALSE);
  g_return_val_if_fail (custom_props != NULL, FALSE);

  tokens = g_strsplit (custom_props, ",", -1);
  for (i = 0; tokens[i]; i++) {
    gchar **kv = g_strsplit (tokens[i], ":", 2);

    if (g_strv_length (kv) == 2) {
      const gchar *key = g_strstrip (kv[0]);
      const gchar *value = g_strstrip (kv[1]);

      if (*key && *value) {
        int ret = nns_edge_set_info (edge_h, key, value);

        if (NNS_EDGE_ERROR_NONE != ret) {
          nns_logw ("Failed to set custom property '%s:%s' (error %d).",
              key, value, ret);
          all_set = FALSE;
        }
      } else {
        nns_logw ("Ignored custom property token '%s' with empty key or value.",
            tokens[i]);
        all_set = FALSE;
      }
    } else if (kv[0] && *g_strstrip (kv[0])) {
      nns_logw ("Ignored malformed custom property token '%s'. "
          "Expected key:value form.", tokens[i]);
      all_set = FALSE;
    }
    g_strfreev (kv);
  }
  g_strfreev (tokens);

  return all_set;
}
