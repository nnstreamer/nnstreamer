/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.c
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @author Junhwan Kim <jejudo.kim@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tensor_query_common.h"

#ifndef EREMOTEIO
#define EREMOTEIO 121           /* This is Linux-specific. Define this for non-Linux systems */
#endif

/**
 * @brief register GEnumValue array for query protocol property handling
 */
GType
gst_tensor_query_protocol_get_type (void)
{
  static GType protocol = 0;
  if (protocol == 0) {
    static GEnumValue protocols[] = {
      {NNS_EDGE_PROTOCOL_TCP, "TCP",
          "Raw TCP protocol. Directly sending stream frames via TCP connections."},
      {NNS_EDGE_PROTOCOL_UDP, "UDP",
          "Raw UDP protocol. Directly sending stream frames via UDP connections."},
      {NNS_EDGE_PROTOCOL_MQTT, "MQTT", "Connect with MQTT brokers."},
      {0, NULL, NULL},
    };
    protocol = g_enum_register_static ("tensor_query_protocol", protocols);
  }

  return protocol;
}
