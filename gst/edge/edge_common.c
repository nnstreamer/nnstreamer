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
          /** @todo support UDP, MQTT and HYBRID */
      {0, NULL, NULL},
    };
    protocol = g_enum_register_static ("edge_protocol", protocols);
  }

  return protocol;
}
